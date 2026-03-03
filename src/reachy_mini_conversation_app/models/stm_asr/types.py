from dataclasses import dataclass, field
from typing import List, Tuple


WordTiming = Tuple[float, float, str]


@dataclass
class TranscriptionResult:
    """Result from a transcription request.

    :param key: Identifier for the segment that was transcribed
    :param words: List of (start, end, word) tuples
    :param text: Concatenated text (convenience)
    :param is_final: Whether this is a final transcription
    :param audio_duration: Duration of the audio buffer in seconds
    :param transcription_time: Wall-clock time the transcription took
    """

    key: int
    words: List[WordTiming] = field(default_factory=list)
    text: str = ""
    is_final: bool = False
    audio_duration: float = 0.0
    transcription_time: float = 0.0
    buffer_offset: float = 0.0  # absolute start time of the audio buffer


@dataclass
class TranscriberStats:
    """Statistics from the async transcriber."""

    queue_size: int = 0
    processed: int = 0
    dropped: int = 0
