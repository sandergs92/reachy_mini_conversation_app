from typing import List, Tuple

import numpy as np
from faster_whisper import WhisperModel

from .config import FasterWhisperConf

# (start, end, word)
WordTiming = Tuple[float, float, str]


class FasterWhisperASR:
    def __init__(self, params: FasterWhisperConf = None, num_workers: int = 1):
        self.params = params if params is not None else FasterWhisperConf()
        self.model = WhisperModel(
            self.params.model_size,
            device=self.params.device,
            compute_type=self.params.compute_type,
            num_workers=num_workers,
        )

    def transcribe_segment(self, audio_samples: np.ndarray) -> List[WordTiming]:
        """Transcribe audio and return word-level timestamps.

        Returns:
            List of (start, end, word) tuples.
        """
        if audio_samples.dtype == np.int16:
            audio_float = audio_samples.astype(np.float32) / 32768.0
        else:
            audio_float = audio_samples.astype(np.float32)

        segments, _info = self.model.transcribe(
            audio_float,
            language=self.params.language,
            beam_size=self.params.beam_size,
            condition_on_previous_text=self.params.condition_on_previous_text,
            vad_filter=False,
            word_timestamps=True,
        )

        words = []
        for seg in segments:
            if seg.words:
                for w in seg.words:
                    words.append((w.start, w.end, w.word))

        return words
