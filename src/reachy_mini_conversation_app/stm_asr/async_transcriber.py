import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from .asr import FasterWhisperASR
from .config import FasterWhisperConf
from .types import TranscriberStats, TranscriptionResult

SAMPLE_RATE = 16000


class AsyncTranscriber:
    def __init__(
        self,
        config,  # FasterWhisperConf or FasterWhisperASR
        num_workers: int = 1,
    ):
        if isinstance(config, FasterWhisperConf):
            self._params = config
            self._asr = FasterWhisperASR(config, num_workers=num_workers)
        else:
            self._params = config.params
            self._asr = config

        self._num_workers = num_workers
        self._result_queue = queue.Queue()
        self._processed_count = 0
        self._pending_count = 0
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=num_workers)

    def _transcribe_and_enqueue(self, key, audio, is_final, buffer_offset):
        """Run transcription and put result in queue."""
        buf_duration = len(audio) / SAMPLE_RATE
        t0 = time.monotonic()
        try:
            words = self._asr.transcribe_segment(audio)
        except Exception:
            words = []
        elapsed = time.monotonic() - t0
        text = " ".join(w[2] for w in words).strip()
        self._result_queue.put(
            TranscriptionResult(
                key=key,
                words=words,
                text=text,
                is_final=is_final,
                audio_duration=buf_duration,
                transcription_time=elapsed,
                buffer_offset=buffer_offset,
            )
        )
        with self._lock:
            self._pending_count -= 1

    def request_transcription(self, key, audio, is_final=False, buffer_offset=0.0):
        with self._lock:
            self._pending_count += 1
        self._pool.submit(self._transcribe_and_enqueue, key, audio, is_final, buffer_offset)

    def get_completed(self):
        results = []
        while True:
            try:
                result = self._result_queue.get_nowait()
                self._processed_count += 1
                results.append(result)
            except queue.Empty:
                break
        return results

    def is_idle(self):
        with self._lock:
            return self._pending_count == 0 and self._result_queue.empty()

    def get_stats(self) -> TranscriberStats:
        with self._lock:
            pending = self._pending_count
        return TranscriberStats(
            queue_size=pending,
            processed=self._processed_count,
            dropped=0,
        )

    def shutdown(self):
        self._pool.shutdown(wait=True)
