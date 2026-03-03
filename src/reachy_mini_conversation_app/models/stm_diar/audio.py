"""Audio preprocessing: raw waveform → mel-spectrogram features."""

from __future__ import annotations

import numpy as np
import torch


class AudioProcessor:
    """Converts raw float32 audio into model-ready mel-spectrogram tensors."""

    def __init__(self, preprocessor, device: torch.device) -> None:
        """
        Args:
            preprocessor: The NeMo model's ``preprocessor`` attribute
                          (``diar_model.preprocessor``).
            device:       Target torch device (cuda / cpu).
        """
        self.preprocessor = preprocessor
        self.device = device

        self.sr: int = 16_000
        self.hop_s: float = float(preprocessor._cfg.window_stride)
        self.hop: int = int(round(self.sr * self.hop_s))

    # ------------------------------------------------------------------ #
    def melspectrogram(
        self, raw_audio: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert a 1-D float32 waveform into (processed_signal, length).

        Returns:
            processed_signal:        ``(1, T, F)``  mel features.
            processed_signal_length: ``(1,)``        valid frame count.
        """
        audio_signal = torch.tensor(
            raw_audio, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        audio_signal_length = torch.tensor(
            [audio_signal.shape[1]], device=self.device
        )

        processed_signal, processed_signal_length = self.preprocessor(
            input_signal=audio_signal, length=audio_signal_length
        )
        # (B, F, T) → (B, T, F)
        processed_signal = processed_signal.transpose(1, 2)
        T_valid = int(processed_signal_length.item())
        processed_signal = processed_signal[:, :T_valid, :]
        return processed_signal, processed_signal_length
