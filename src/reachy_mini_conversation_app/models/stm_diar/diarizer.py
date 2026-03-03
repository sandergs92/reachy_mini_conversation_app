"""High-level streaming diarizer built on NVIDIA Streaming Sortformer 4spk-v2."""

from __future__ import annotations
from pathlib import Path

import numpy as np
import torch
from nemo.collections.asr.models import SortformerEncLabelModel
from nemo.collections.asr.parts.utils.vad_utils import (
    ts_vad_post_processing,
    load_postprocessing_from_yaml,
)

from reachy_mini_conversation_app.models.stm_diar.audio import AudioProcessor
from reachy_mini_conversation_app.models.stm_diar.types import DiarSegment
from reachy_mini_conversation_app.models.stm_diar.config import (
    DEFAULT_NEMO,
    DEFAULT_ONSET,
    DEFAULT_PP_YAML,
    FRAME_DURATION_S,
    StreamingConfig,
)


class StreamingDiarizer:
    """Wrap Streaming Sortformer for chunk-by-chunk numpy-in / labels-out use.

    Usage::

        diar = StreamingDiarizer.from_nemo()
        diar.reset()

        for chunk in audio_chunks:
            active = diar.step(chunk)       # dict[int, (start, end)]
            final  = diar.collect_final()   # list[DiarSegment] — newly ended

        segments = diar.finalize(duration)  # full post-processed output
    """

    def __init__(
        self,
        model: SortformerEncLabelModel,
        device: torch.device,
        config: StreamingConfig | None = None,
        pp_yaml: str | Path | None = None,
        onset: float = DEFAULT_ONSET,
    ) -> None:
        self.model = model
        self.device = device
        self.config = config or StreamingConfig()
        self.audio_proc = AudioProcessor(model.preprocessor, device)
        self._pp_yaml = str(pp_yaml or DEFAULT_PP_YAML)
        self._onset = onset

        self._apply_streaming_config()
        self.model.streaming_mode = True

        # Post-processing config
        self._pp_cfg = load_postprocessing_from_yaml(self._pp_yaml)
        self._unit_10ms = int(model._cfg.encoder.subsampling_factor)

        # Audio-level chunking params (matching reference implementation)
        sm = self.model.sortformer_modules
        hop = self.audio_proc.hop
        self._subfac = sm.subsampling_factor
        self._base_frames = sm.chunk_len * self._subfac
        self._chunk_size = self._base_frames * hop

        # Left/right context in mel frames and audio samples
        self._L_off = 8
        self._R_off = 8
        self._ctx_left = self._L_off * hop
        self._ctx_right = self._R_off * hop

        # Real-time turn tracking
        self._frame_offset: int = 0
        self._active: dict[int, float] = {}
        self._pending_final: list[DiarSegment] = []

    # ── Construction helpers ────────────────────────────────────────
    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "nvidia/diar_streaming_sortformer_4spk-v2",
        *,
        config: StreamingConfig | None = None,
        pp_yaml: str | Path | None = None,
        device: torch.device | str | None = None,
    ) -> StreamingDiarizer:
        """Download & load from Hugging Face."""
        device = _resolve_device(device)
        model = SortformerEncLabelModel.from_pretrained(model_name)
        model = model.to(device).eval()
        return cls(model, device, config, pp_yaml)

    @classmethod
    def from_nemo(
        cls,
        nemo_path: str | Path | None = None,
        *,
        config: StreamingConfig | None = None,
        pp_yaml: str | Path | None = None,
        device: torch.device | str | None = None,
    ) -> StreamingDiarizer:
        """Load from a local ``.nemo`` checkpoint."""
        device = _resolve_device(device)
        nemo_path = str(nemo_path or DEFAULT_NEMO)
        model = SortformerEncLabelModel.restore_from(
            restore_path=nemo_path,
            map_location=device,
            strict=False,
        )
        model = model.to(device).eval()
        return cls(model, device, config, pp_yaml)

    # ── Streaming API ───────────────────────────────────────────────
    @property
    def chunk_samples(self) -> int:
        """Audio samples the caller should feed per step."""
        return self._chunk_size

    def reset(self) -> None:
        """Clear all state for a new session."""
        self._frame_offset = 0
        self._active.clear()
        self._pending_final.clear()

        self._streaming_state = self.model.sortformer_modules.init_streaming_state(
            batch_size=1, async_streaming=True, device=self.device
        )
        self._total_preds = torch.zeros((1, 0, self.model.sortformer_modules.n_spk), device=self.device)

        # Audio-level buffer (raw float32 samples)
        self._audio_buffer = np.zeros(0, dtype=np.float32)
        self._buffer_offset = 0
        self._processed_until = 0

    def step(self, chunk: np.ndarray) -> dict[int, tuple[float, float]]:
        """Feed one audio chunk and return currently-active speakers.

        Args:
            chunk: 1-D float32 array of audio samples.

        Returns:
            ``{speaker_id: (start_time, current_end_time)}`` for every
            speaker that is active at the end of this chunk.
            Concluded turns are available via :meth:`collect_final`.

        """
        self._audio_buffer = np.concatenate([self._audio_buffer, chunk])

        any_new_preds = False

        while True:
            base_start = self._processed_until
            base_end = base_start + self._chunk_size

            needed_end = base_end + self._ctx_right
            available_end = self._buffer_offset + len(self._audio_buffer)

            if needed_end > available_end:
                break

            if base_start == 0:
                start_global = 0
            else:
                start_global = max(0, base_start - self._ctx_left)

            end_global = min(available_end, needed_end)

            local_start = start_global - self._buffer_offset
            local_end = end_global - self._buffer_offset
            raw_with_ctx = self._audio_buffer[local_start:local_end]

            processed_signal, processed_signal_length = self.audio_proc.melspectrogram(raw_with_ctx)

            with torch.inference_mode(), torch.amp.autocast(self.device.type, enabled=(self.device.type == "cuda")):
                self._streaming_state, self._total_preds = self.model.forward_streaming_step(
                    processed_signal=processed_signal,
                    processed_signal_length=processed_signal_length,
                    streaming_state=self._streaming_state,
                    total_preds=self._total_preds,
                    left_offset=(0 if base_start == 0 else self._L_off),
                    right_offset=self._R_off,
                )

            self._processed_until = base_end
            any_new_preds = True

            next_start = max(0, self._processed_until - self._ctx_left)
            safe_drop = min(
                max(0, next_start - self._buffer_offset),
                len(self._audio_buffer),
            )
            if safe_drop > 0:
                self._audio_buffer = self._audio_buffer[safe_drop:]
                self._buffer_offset += safe_drop

        if not any_new_preds:
            return {spk: (start, self._frame_offset * FRAME_DURATION_S) for spk, start in self._active.items()}

        # ── Real-time binarisation from total_preds ─────────────────
        total_frames = self._total_preds.shape[1]
        new_frames = total_frames - self._frame_offset

        if new_frames <= 0:
            return {spk: (start, self._frame_offset * FRAME_DURATION_S) for spk, start in self._active.items()}

        chunk_preds = self._total_preds[0, self._frame_offset :].float().cpu().numpy()
        T = chunk_preds.shape[0]
        num_spk = chunk_preds.shape[1]

        chunk_end_time = (self._frame_offset + T) * FRAME_DURATION_S

        active_now: set[int] = set()
        for spk in range(num_spk):
            if (chunk_preds[:, spk] >= self._onset).any():
                active_now.add(spk)

        for spk in list(self._active):
            if spk not in active_now:
                seg = DiarSegment(
                    speaker=spk,
                    start=self._active.pop(spk),
                    end=self._frame_offset * FRAME_DURATION_S,
                )
                self._pending_final.append(seg)

        for spk in active_now:
            if spk not in self._active:
                self._active[spk] = self._frame_offset * FRAME_DURATION_S

        self._frame_offset += T

        return {spk: (start, chunk_end_time) for spk, start in self._active.items()}

    def collect_final(self) -> list[DiarSegment]:
        """Pop and return turns that concluded since the last call."""
        out = self._pending_final.copy()
        self._pending_final.clear()
        return out

    def get_speaker_timestamps(self, base_time: float = 0.0) -> list:
        """Convert total_preds to post-processed timestamp intervals per speaker."""
        probs = self._total_preds[0].detach().cpu()
        num_speakers = probs.shape[1]

        speaker_timestamps = []
        for spk_id in range(num_speakers):
            spk_probs = probs[:, spk_id]
            ts_mat = ts_vad_post_processing(
                spk_probs,
                cfg_vad_params=self._pp_cfg,
                unit_10ms_frame_count=self._unit_10ms,
                bypass_postprocessing=False,
            )
            ts_list = [[base_time + float(stt), base_time + float(end)] for (stt, end) in ts_mat.tolist()]
            speaker_timestamps.append(ts_list)
        return speaker_timestamps

    def finalize(self, audio_duration: float) -> list[DiarSegment]:
        """Post-process total_preds and return final segments."""
        speaker_timestamps = self.get_speaker_timestamps()

        segments: list[DiarSegment] = []
        for spk_idx, ts_list in enumerate(speaker_timestamps):
            for seg_start, seg_end in ts_list:
                if seg_end <= audio_duration:
                    segments.append(DiarSegment(spk_idx, seg_start, seg_end))

        segments.sort(key=lambda s: s.start)
        return segments

    # ── Internals ───────────────────────────────────────────────────
    def _apply_streaming_config(self) -> None:
        c = self.config
        sm = self.model.sortformer_modules
        sm.chunk_len = c.chunk_len
        sm.chunk_right_context = c.chunk_right_context
        sm.fifo_len = c.fifo_len
        sm.spkcache_update_period = c.spkcache_update_period
        sm.spkcache_len = c.spkcache_len
        sm.log = False
        sm._check_streaming_parameters()


# ── Helpers ─────────────────────────────────────────────────────────────
def _resolve_device(device: torch.device | str | None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)
