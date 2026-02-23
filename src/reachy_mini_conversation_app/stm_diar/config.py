"""Configuration for the streaming diarizer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# ── Package-level paths ─────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "data"
DEFAULT_PP_YAML = DATA_DIR / "diar_streaming_sortformer_4spk-v2_callhome-part1.yaml"
DEFAULT_NEMO = DATA_DIR / "diar_streaming_sortformer_4spk-v2.nemo"

# Sortformer outputs one frame per 80 ms (8 × 10 ms sub-frames)
FRAME_DURATION_S = 0.08
DEFAULT_ONSET = 0.5


@dataclass
class StreamingConfig:
    """Ultra-low-latency streaming parameters (0.32s latency, RTF ≈ 0.18).

    Latency = (chunk_len + chunk_right_context) * 80ms = 0.32s.
    """

    chunk_len: int = 3
    chunk_right_context: int = 1
    fifo_len: int = 188
    spkcache_update_period: int = 144
    spkcache_len: int = 188
