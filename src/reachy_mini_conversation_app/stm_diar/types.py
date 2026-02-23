"""Shared types for the streaming diarizer."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DiarSegment:
    """A single speaker-homogeneous segment."""

    speaker: int
    start: float
    end: float

    def __repr__(self) -> str:
        return f"spk_{self.speaker}  {self.start:7.2f}s – {self.end:7.2f}s"
