import time
import atexit
import subprocess
import multiprocessing as mp
from queue import Empty, Queue
from typing import Any, Literal, Optional
from contextlib import contextmanager
from collections import Counter, deque
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory

import numpy as np

from reachy_mini_conversation_app.display import (
    DIM,
    BOLD,
    RESET,
    StreamingDisplay,
    tag_words,
)
from reachy_mini_conversation_app.stm_asr import (
    AsyncTranscriber,
    HypothesisBuffer,
    FasterWhisperConf,
)
from reachy_mini_conversation_app.stm_diar.config import FRAME_DURATION_S


@contextmanager
def cuda_mps():
    """Start CUDA MPS daemon on entry, stop on exit.

    MPS lets multiple processes share the GPU without context-switch overhead.
    Requires that no other CUDA contexts are active on the GPU.
    Falls back silently if MPS is already running or can't be started.
    """
    started = False
    try:
        # Check if MPS is already running
        result = subprocess.run(
            ["nvidia-cuda-mps-control", "-d"],
            capture_output=True,
            timeout=5,
        )
        started = result.returncode == 0
        if started:
            print(f"{DIM}CUDA MPS daemon started{RESET}")
        else:
            # Might already be running
            print(f"{DIM}CUDA MPS already active or unavailable{RESET}")
    except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError):
        print(f"{DIM}CUDA MPS not available, continuing without it{RESET}")

    try:
        yield
    finally:
        if started:
            try:
                subprocess.run(
                    ["bash", "-c", "echo quit | nvidia-cuda-mps-control"],
                    capture_output=True,
                    timeout=5,
                )
                print(f"{DIM}CUDA MPS daemon stopped{RESET}")
            except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError):
                pass


# ── Diarizer process ────────────────────────────────────────────────────
def _diar_worker(
    audio_queue: mp.Queue,
    ready_event,
    preds_shm_name: str,
    preds_shape: tuple,
    preds_len: mp.Value,
    max_frames: int,
    num_spk: int,
):
    """Runs in a separate process with its own CUDA context."""
    from multiprocessing.shared_memory import SharedMemory

    import torch

    from reachy_mini_conversation_app.stm_diar import StreamingDiarizer

    diar = StreamingDiarizer.from_nemo()
    diar.reset()

    shm = SharedMemory(name=preds_shm_name, create=False)
    preds_buf = np.ndarray((max_frames, num_spk), dtype=np.float32, buffer=shm.buf)

    # Warmup: run a dummy chunk to trigger CUDA kernel compilation
    dummy_chunk_size = diar.chunk_samples
    with (
        torch.inference_mode(),
        torch.amp.autocast(diar.device.type, enabled=(diar.device.type == "cuda")),
    ):
        _DIAR_LOOKAHEAD_SAMPLES = 1280  # NeMo MSDD lookahead: 2 × 80ms @ 8kHz
        dummy = np.zeros(dummy_chunk_size + 2 * _DIAR_LOOKAHEAD_SAMPLES, dtype=np.float32)
        diar.step(dummy)
    diar.reset()
    preds_len.value = 0
    ready_event.set()

    with (
        torch.inference_mode(),
        torch.amp.autocast(diar.device.type, enabled=(diar.device.type == "cuda")),
    ):
        while True:
            chunk = audio_queue.get()
            if chunk is None:
                break
            diar.step(chunk)

            # Copy preds to shared memory
            tp = diar._total_preds
            if tp.shape[1] > 0:
                snap = tp[0].float().cpu().numpy()
                n = min(snap.shape[0], max_frames)
                preds_buf[:n, :] = snap[:n, :]
                preds_len.value = n

    shm.close()


# ── Speaker lookup from shared memory ───────────────────────────────────
class SpeakerLookup:
    """Reads diarizer predictions from shared memory (written by diar process)."""

    def __init__(
        self,
        shm_name: str,
        max_frames: int,
        num_spk: int,
        preds_len: mp.Value,
        threshold: float = 0.3,
    ):
        from multiprocessing.shared_memory import SharedMemory

        self._shm = SharedMemory(name=shm_name, create=False)
        atexit.register(lambda: SpeakerLookup._safe_shm_cleanup(self._shm))
        self._buf = np.ndarray((max_frames, num_spk), dtype=np.float32, buffer=self._shm.buf)
        self._preds_len = preds_len
        self._threshold = threshold

    def get_speaker(self, timestamp: float) -> int | None:
        # NOTE: preds_len and the shared buffer are written by a separate process.
        # We read preds_len first, then copy one row. The copy() mitigates torn
        # reads for that row. There is no explicit barrier, but in practice
        # mp.Value access serializes via a lock, and np.ndarray.copy() is atomic
        # at the row level on x86. If you ever port to a platform without TSO
        # guarantees, add an explicit mp.Lock here.
        n = self._preds_len.value
        if n == 0:
            return None
        frame_idx = int(timestamp / FRAME_DURATION_S)
        frame_idx = max(0, min(frame_idx, n - 1))
        probs = self._buf[frame_idx].copy()
        if probs.max() < self._threshold:
            return None
        return int(probs.argmax())

    def get_word_speaker(self, word) -> int | None:
        mid = (word[0] + word[1]) / 2
        return self.get_speaker(mid)

    def close(self):
        self._shm.close()

    @staticmethod
    def _safe_shm_cleanup(shm: SharedMemory) -> None:
        try:
            shm.close()
        except Exception:
            pass
        try:
            shm.unlink()
        except Exception:
            pass


@dataclass
class DiarAsrPipeline:
    async_asr: AsyncTranscriber
    speaker_lookup: "SpeakerLookup"

    # diarizer infra (needed for feeding + shutdown)
    audio_queue: mp.Queue
    diar_proc: mp.Process

    # shared memory ownership
    shm: SharedMemory
    preds_len: mp.Value

    # bookkeeping (optional but handy)
    max_frames: int
    num_spk: int

    def close(self):
        """Graceful shutdown + free shared memory."""
        try:
            self.audio_queue.put(None)
        except Exception:
            pass

        try:
            self.diar_proc.join(timeout=10)
            if self.diar_proc.is_alive():
                self.diar_proc.terminate()
                self.diar_proc.join(timeout=3)
        except Exception:
            pass

        try:
            self.shm.close()
        except Exception:
            pass

        try:
            self.shm.unlink()
        except Exception:
            pass

        try:
            self.speaker_lookup.close()
        except Exception:
            pass

        try:
            self.async_asr.shutdown()
        except Exception:
            pass


def _safe_shm_cleanup(shm: SharedMemory) -> None:
    try:
        shm.close()
    except Exception:
        pass
    try:
        shm.unlink()
    except Exception:
        pass


def build_diar_asr_pipeline(
    *,
    max_duration_s: float = 3600.0,
    language: str = "nl",
    num_spk: int = 4,
    asr_model_size: str = "turbo",
    asr_device: str = "cuda",
    asr_compute_type: str = "int8_bfloat16",
    asr_num_workers: int = 2,
    asr_beam_size: int = 5,
    condition_on_previous_text: bool = False,
    diar_ready_timeout_s: float = 30.0,
) -> DiarAsrPipeline:
    """Create + warm up ASR and streaming diarizer process, return ready-to-use objects."""
    # ---- ASR ----
    asr_conf = FasterWhisperConf(
        model_size=asr_model_size,
        device=asr_device,
        compute_type=asr_compute_type,
        language=language,
        beam_size=asr_beam_size,
        condition_on_previous_text=condition_on_previous_text,
    )
    async_asr = AsyncTranscriber(asr_conf, num_workers=asr_num_workers)

    # ---- Diarizer process + shared memory ----
    max_frames = int(max_duration_s / FRAME_DURATION_S) + 1000

    ctx = mp.get_context("spawn")
    shm = SharedMemory(
        create=True,
        size=max_frames * num_spk * np.dtype(np.float32).itemsize,
    )
    atexit.register(lambda: _safe_shm_cleanup(shm))
    preds_len = ctx.Value("i", 0)

    audio_queue = ctx.Queue(maxsize=64)
    diar_ready = ctx.Event()

    diar_proc = ctx.Process(
        target=_diar_worker,
        args=(
            audio_queue,
            diar_ready,
            shm.name,
            (max_frames, num_spk),
            preds_len,
            max_frames,
            num_spk,
        ),
        daemon=True,
    )
    diar_proc.start()

    # ---- Warmups ----
    # Warmup ASR: compile CUDA kernels etc.
    dummy_audio = np.zeros(16000, dtype=np.float32)  # 1s silence @16kHz
    async_asr.request_transcription(-1, dummy_audio, is_final=True, buffer_offset=0.0)
    while not async_asr.is_idle():
        async_asr.get_completed()
        time.sleep(0.01)
    async_asr.get_completed()  # drain dummy result

    # Wait for diarizer warmup
    if not diar_ready.wait(timeout=diar_ready_timeout_s):
        diar_proc.terminate()
        shm.unlink()
        raise RuntimeError(f"Diarizer process failed to warm up within {diar_ready_timeout_s}s")

    speaker_lookup = SpeakerLookup(shm.name, max_frames, num_spk, preds_len)

    return DiarAsrPipeline(
        async_asr=async_asr,
        speaker_lookup=speaker_lookup,
        audio_queue=audio_queue,
        diar_proc=diar_proc,
        shm=shm,
        preds_len=preds_len,
        max_frames=max_frames,
        num_spk=num_spk,
    )


@dataclass
class HypothesisEvent:
    """Hypothesis updates emitted by the streaming coordinator.

    kind='active' changes frequently (partial hypothesis).
    kind='confirmed' is committed text (stable).
    """

    kind: Literal["active", "confirmed", "active_turn", "confirmed_turn"]
    text: str
    words: list
    t0: float
    t1: float
    is_final: bool = False
    meta: Optional[dict[str, Any]] = None


class StreamingCoordinator:
    def __init__(
        self,
        preds_len,
        pending_words,
        known_total_duration_s: float | None,
        display,
        speaker_lookup,
        rtf_values,
        chunk_samples,
        sample_rate,
        hyp_buffer,
        event_queue: Queue | None = None,
        stream_wallclock_t0: float | None = None,
        confirmed_latency_values: list[float] | None = None,
    ):
        self.preds_len = preds_len
        self.pending_words = pending_words
        self.known_total_duration_s = known_total_duration_s
        self.stream_ended = False
        self.stream_end_time_s: float | None = None
        self.display = display
        self.speaker_lookup = speaker_lookup
        self.total_processed = 0
        self.rtf_values = rtf_values
        self.chunk_samples = chunk_samples
        self.sample_rate = sample_rate
        self.hyp_buffer = hyp_buffer
        self.next_expected_key = 0
        self.reorder_buf = {}
        self.event_queue = event_queue
        self._last_active_text = ""
        self.stream_wallclock_t0 = stream_wallclock_t0
        self.confirmed_latency_values = confirmed_latency_values if confirmed_latency_values is not None else []
        # ── Dual-lane turn aggregation ───────────────────────────────────
        # Turn boundary is ONLY a dominant speaker change.
        self._turn_id = 0
        self._turn_speaker: int | None = None
        self._confirmed_words: list = []
        self._last_prefix: str = ""

    def _dominant_speaker(self, words: list) -> int | None:
        """Dominant speaker id over these words (None if unknown)."""
        if not words:
            return None
        c: Counter[int] = Counter()
        for w in words:
            spk = self.speaker_lookup.get_word_speaker(w)
            if spk is not None:
                c[int(spk)] += 1
        if not c:
            return None
        return c.most_common(1)[0][0]

    def _flush_turn(self, *, is_final: bool = False, prefix: str | None = None) -> None:
        """Emit aggregated confirmed_turn event for the current speaker and clear buffers."""
        spk = self._turn_speaker
        turn_id = self._turn_id
        prefix = prefix or self._last_prefix  # explicit wins over stale _last_prefix
        confirmed_words = self._confirmed_words  # capture reference before reset

        # Reset buffers first — we captured references above so these rebinds
        # don't affect the local variables; this prevents accidental reuse if
        # _flush_turn is somehow re-entered.
        self._confirmed_words = []

        if self.event_queue is None or spk is None:
            return

        if confirmed_words:
            t0 = float(confirmed_words[0][0])
            t1 = float(confirmed_words[-1][1])
            conf_text = " ".join(w[2] for w in confirmed_words).strip()
            if conf_text:
                self.event_queue.put(
                    HypothesisEvent(
                        kind="confirmed_turn",
                        text=conf_text,
                        words=confirmed_words,
                        t0=t0,
                        t1=t1,
                        is_final=is_final,
                        meta={"speaker": spk, "turn_id": turn_id, "prefix": prefix},
                    )
                )

    def _maybe_switch_turn(self, spk: int | None) -> None:
        """Switch turn ONLY when speaker changes; flush previous turn immediately."""
        if spk is None:
            return
        if self._turn_speaker is None:
            self._turn_speaker = spk
            self._turn_id += 1
            return
        if spk != self._turn_speaker:
            self._flush_turn(is_final=False)
            self._turn_speaker = spk
            self._turn_id += 1

    def flush_current_turn(self, *, is_final: bool = True) -> None:
        """Public flush (end-of-stream)."""
        self._flush_turn(is_final=is_final)

    def _emit_confirmed(self, prefix: str, words: list, *, is_final: bool = False) -> None:
        if self.event_queue is None or not words:
            return

        # Emit one confirmed event per word (unchanged), but accumulate
        # confirmed_words per-speaker so turn boundaries are respected.
        t0 = float(words[0][0])
        t1 = float(words[-1][1])
        text = " ".join(w[2] for w in words)
        self.event_queue.put(
            HypothesisEvent(
                kind="confirmed",
                text=text,
                words=words,
                t0=t0,
                t1=t1,
                is_final=is_final,
                meta={"prefix": prefix},
            )
        )

        # Walk word-by-word, switching turns when speaker changes, so that
        # words are always accumulated into the correct speaker's turn buffer.
        for w in words:
            spk = self.speaker_lookup.get_word_speaker(w)
            if spk is None:
                # Unknown speaker: attach to current turn if one exists
                if self._turn_speaker is not None:
                    self._confirmed_words.append(w)
                continue
            self._maybe_switch_turn(spk)
            if spk == self._turn_speaker:
                self._confirmed_words.append(w)

    def _emit_active(self, prefix: str, words: list, *, chunk_key: int) -> None:
        if self.event_queue is None or not words:
            return
        text = " ".join(w[2] for w in words)
        if text == self._last_active_text:
            return
        self._last_active_text = text
        active_spk = self._dominant_speaker(words)
        t0 = float(words[0][0])
        t1 = float(words[-1][1])
        self.event_queue.put(
            HypothesisEvent(
                kind="active",
                text=text,
                words=words,
                t0=t0,
                t1=t1,
                meta={"prefix": prefix, "speaker": active_spk},
            )
        )

    def mark_stream_end(self, stream_end_time_s: float):
        self.stream_ended = True
        self.stream_end_time_s = stream_end_time_s

    def _flush_pending(self):
        """Display pending words whose timestamps are now covered by diarizer."""
        diar_time = self.preds_len.value * FRAME_DURATION_S
        while self.pending_words:
            prefix, words = self.pending_words[0]
            last_word_time = words[-1][1] if words else 0.0

            end_guard = (
                self.stream_ended
                and self.stream_end_time_s is not None
                and diar_time >= (self.stream_end_time_s - 1.0)
            )

            if last_word_time <= diar_time or end_guard:
                self.pending_words.popleft()
                self.display.print_final(
                    f"{prefix} {tag_words(words, self.speaker_lookup, underline_words=False, dim_words=False)}{RESET}"
                )
            else:
                break

    def _handle_result(self, result):
        self.total_processed += 1
        rtf = result.transcription_time / result.audio_duration if result.audio_duration > 0 else 0
        self.rtf_values.append(rtf)

        current_time = (result.key + self.chunk_samples) / self.sample_rate

        if self.known_total_duration_s:
            progress = min(100.0, (current_time / self.known_total_duration_s) * 100.0)
            prefix = f"[{progress:6.2f}% | RTF {rtf:.2f}x]"
        else:
            prefix = f"[{current_time:7.2f}s | RTF {rtf:.2f}x]"
        self._last_prefix = prefix

        self.hyp_buffer.insert(result.words, offset=result.buffer_offset)
        committed = self.hyp_buffer.flush()

        if committed:
            # Approximate confirmed-chunk latency (wall-clock now minus audio time of last word)
            if self.stream_wallclock_t0 is not None:
                t1 = float(committed[-1][1])
                lat_s = time.perf_counter() - (self.stream_wallclock_t0 + t1)
                self.confirmed_latency_values.append(lat_s)
            self.pending_words.append((prefix, committed))
            self._emit_confirmed(prefix, committed, is_final=False)

        # Try to flush any pending words the diarizer has caught up to
        self._flush_pending()

        active = self.hyp_buffer.complete()
        if active:
            self._emit_active(prefix, active, chunk_key=result.key)
            self.display.update_active(
                f"{prefix} {tag_words(active, self.speaker_lookup, underline_words=False, dim_words=True)}{RESET}"
            )

    def confirmed_latency_stats(self) -> dict | None:
        """Return latency stats for confirmed chunks (seconds), or None if empty."""
        if not self.confirmed_latency_values:
            return None
        arr = np.asarray(self.confirmed_latency_values, dtype=float)
        return {
            "count": int(arr.size),
            "avg_s": float(arr.mean()),
            "p95_s": float(np.percentile(arr, 95)),
            "max_s": float(arr.max()),
            "min_s": float(arr.min()),
        }

    def _flush_reorder_buf(self):
        while self.next_expected_key in self.reorder_buf:
            result = self.reorder_buf.pop(self.next_expected_key)
            self._handle_result(result)
            self.next_expected_key = result.key + self.chunk_samples


class StreamingDiarAsrStreamer:
    """Chunk-in / text-out streaming API.

    Usage:
        pipe = build_diar_asr_pipeline(...)
        streamer = StreamingDiarAsrStreamer(pipe, ...)
        for key, chunk, is_last in ...:
            streamer.push(key, chunk, is_final=is_last)
        streamer.finalize()
        pipe.close()
    """

    def __init__(
        self,
        pipe: DiarAsrPipeline,
        *,
        sample_rate: int,
        known_total_duration_s: float | None = None,
        chunk_samples: int,
        diar_chunk_samples: int,
        max_seconds: float = 10.0,
        display: StreamingDisplay | None = None,
        hyp_buffer: HypothesisBuffer | None = None,
    ):
        self.pipe = pipe
        self.sample_rate = sample_rate
        self.known_total_duration_s = known_total_duration_s
        self.audio_seen_s = 0.0
        self.chunk_samples = chunk_samples
        self.diar_chunk_samples = diar_chunk_samples

        self.display = display or StreamingDisplay()
        self.hyp_buffer = hyp_buffer or HypothesisBuffer()

        chunk_duration = chunk_samples / sample_rate
        max_chunks = max(1, int(max_seconds / chunk_duration))
        self.audio_buffer = deque(maxlen=max_chunks)

        self.total_submitted = 0
        self.rtf_values: list[float] = []

        # Pre-allocated diar accumulator buffer (avoids O(n²) concatenation)
        _diar_buf_size = diar_chunk_samples * 8
        self._diar_buf = np.zeros(_diar_buf_size, dtype=np.float32)
        self._diar_buf_len = 0

        # (prefix, words) waiting for diarizer to catch up
        self.pending_words: deque[tuple[str, list]] = deque()

        self._t0 = time.perf_counter()
        self.events: Queue[HypothesisEvent] = Queue()

        self.coordinator = StreamingCoordinator(
            preds_len=self.pipe.preds_len,
            pending_words=self.pending_words,
            known_total_duration_s=self.known_total_duration_s,
            display=self.display,
            speaker_lookup=self.pipe.speaker_lookup,
            rtf_values=self.rtf_values,
            chunk_samples=self.chunk_samples,
            sample_rate=self.sample_rate,
            hyp_buffer=self.hyp_buffer,
            event_queue=self.events,
        )

    def poll_events(self, max_items: int = 100) -> list[HypothesisEvent]:
        """Non-blocking drain of hypothesis events (active + confirmed)."""
        out: list[HypothesisEvent] = []
        for _ in range(max_items):
            try:
                out.append(self.events.get_nowait())
            except Empty:
                break
        return out

    def _drain_completed_once(self) -> bool:
        results = self.pipe.async_asr.get_completed()
        if not results:
            return False
        for result in results:
            self.coordinator.reorder_buf[result.key] = result
        self.coordinator._flush_reorder_buf()
        return True

    def push(self, key: int, chunk: np.ndarray, *, is_final: bool = False) -> None:
        """Push one audio chunk (float32, 16kHz mono) into diarizer + ASR."""
        if chunk.dtype != np.float32:
            chunk = chunk.astype(np.float32)

        # Rolling buffer for ASR context
        self.audio_buffer.append(chunk)

        # Feed diarizer process
        end = self._diar_buf_len + len(chunk)
        if end > len(self._diar_buf):
            # Grow buffer if needed (rare)
            new_buf = np.zeros(end * 2, dtype=np.float32)
            new_buf[: self._diar_buf_len] = self._diar_buf[: self._diar_buf_len]
            self._diar_buf = new_buf
        self._diar_buf[self._diar_buf_len : end] = chunk
        self._diar_buf_len = end

        while self._diar_buf_len >= self.diar_chunk_samples:
            diar_chunk = self._diar_buf[: self.diar_chunk_samples].copy()
            remaining = self._diar_buf_len - self.diar_chunk_samples
            self._diar_buf[:remaining] = self._diar_buf[self.diar_chunk_samples : self._diar_buf_len]
            self._diar_buf_len = remaining
            try:
                self.pipe.audio_queue.put(diar_chunk, timeout=0.5)
            except Exception:
                import warnings

                warnings.warn("Diarizer queue full — dropping chunk; speaker labels may lag.")

        # Feed ASR
        buffer_len = len(self.audio_buffer)
        buffer_end_sample = key + len(chunk)
        self.audio_seen_s = max(self.audio_seen_s, buffer_end_sample / self.sample_rate)
        buffer_start_sample = buffer_end_sample - (buffer_len * self.chunk_samples)
        buffer_offset = max(0.0, buffer_start_sample / self.sample_rate)

        buffer_np = np.concatenate(list(self.audio_buffer)) if buffer_len > 1 else self.audio_buffer[0]

        self.pipe.async_asr.request_transcription(key, buffer_np, is_final=is_final, buffer_offset=buffer_offset)
        self.total_submitted += 1

        if is_final:
            self.coordinator.mark_stream_end(self.audio_seen_s)

        # Drain + flush
        self._drain_completed_once()
        self.coordinator._flush_pending()

    def finalize(self, *, diar_flush_sleep_s: float = 0.5) -> None:
        """Flush remaining diar audio, drain ASR, and force-print remaining pending words."""
        if self._diar_buf_len > 0:
            self.pipe.audio_queue.put(self._diar_buf[: self._diar_buf_len].copy())
            self._diar_buf_len = 0

        while True:
            drained = self._drain_completed_once()
            self.coordinator._flush_pending()

            if (not drained) and self.pipe.async_asr.is_idle():
                # Handle any out-of-order leftovers
                for k in sorted(self.coordinator.reorder_buf.keys()):
                    self.coordinator._handle_result(self.coordinator.reorder_buf[k])
                self.coordinator.reorder_buf.clear()

                # Give diarizer a moment to catch up
                time.sleep(diar_flush_sleep_s)
                self.coordinator._flush_pending()

                # Force-flush anything still pending
                for prefix, words in self.pending_words:
                    self.display.print_final(
                        f"{prefix} {
                            tag_words(words, self.pipe.speaker_lookup, underline_words=False, dim_words=False)
                        }{RESET}"
                    )
                self.pending_words.clear()

                remaining = self.hyp_buffer.complete()
                if remaining:
                    self.display.print_final(
                        f" {tag_words(remaining, self.pipe.speaker_lookup, underline_words=False, dim_words=False)}{
                            RESET
                        }"
                    )
                # Emit any buffered turn-level events for last speaker turn
                self.coordinator.flush_current_turn(is_final=True)
                break

            time.sleep(0.05)
        self.print_results(bold=BOLD, reset=RESET)

    def print_results(self, *, bold: str = "", reset: str = "") -> None:
        elapsed = time.perf_counter() - self._t0
        audio_duration = self.audio_seen_s
        rtf_values = self.rtf_values

        print(f"\n{bold}{'=' * 100}{reset}")
        print(f"{bold}RESULTS{reset}")
        print(f"{bold}{'=' * 100}{reset}")
        print(f"ASR workers:       {self.pipe.async_asr._num_workers}")
        print(f"Audio duration:    {audio_duration:.2f}s")
        print(f"Wall-clock time:   {elapsed:.2f}s")
        print(f"Overall RTF:       {elapsed / audio_duration:.2f}x" if audio_duration else "Overall RTF:       n/a")
        print(f"Submitted:         {self.total_submitted}")
        print(f"Processed:         {self.coordinator.total_processed}")
        if rtf_values:
            print(f"Per-chunk RTF avg: {sum(rtf_values) / len(rtf_values):.3f}x")
            print(f"Per-chunk RTF max: {max(rtf_values):.3f}x")
            print(f"Per-chunk RTF min: {min(rtf_values):.3f}x")
        print(f"{bold}Done!{reset}")
