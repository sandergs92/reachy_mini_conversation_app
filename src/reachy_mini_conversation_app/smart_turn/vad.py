"""Silero VAD wrapper for 16 kHz mono audio (512-sample chunks)."""

import os
import time

import numpy as np
import onnxruntime as ort

_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_SILERO_PATH = os.path.join(_MODEL_DIR, "silero_vad.onnx")


class SileroVAD:
    """Minimal Silero VAD ONNX wrapper for 16 kHz, mono, chunk=512."""

    CHUNK = 512               # Silero expects 512 samples at 16 kHz
    CONTEXT_SIZE = 64         # internal context length
    RESET_INTERVAL_S = 5.0    # reset hidden state periodically

    def __init__(self, model_path: str = _DEFAULT_SILERO_PATH):
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"], sess_options=opts,
        )
        self._state: np.ndarray | None = None
        self._context: np.ndarray | None = None
        self._last_reset_time = time.time()
        self._init_states()

    def _init_states(self):
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, self.CONTEXT_SIZE), dtype=np.float32)

    def maybe_reset(self):
        """Periodically reset hidden state to avoid drift."""
        if (time.time() - self._last_reset_time) >= self.RESET_INTERVAL_S:
            self._init_states()
            self._last_reset_time = time.time()

    def prob(self, chunk_f32: np.ndarray) -> float:
        """
        Compute speech probability for one 512-sample chunk.

        Args:
            chunk_f32: float32 mono audio, exactly 512 samples.

        Returns:
            Speech probability as a float in [0, 1].
        """
        x = np.reshape(chunk_f32, (1, -1))
        if x.shape[1] != self.CHUNK:
            raise ValueError(f"Expected {self.CHUNK} samples, got {x.shape[1]}")
        x = np.concatenate((self._context, x), axis=1)

        ort_inputs = {
            "input": x.astype(np.float32),
            "state": self._state,
            "sr": np.array(16000, dtype=np.int64),
        }
        out, self._state = self.session.run(None, ort_inputs)

        self._context = x[:, -self.CONTEXT_SIZE:]
        self.maybe_reset()
        return float(out[0][0])
