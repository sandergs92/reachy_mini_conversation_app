from dataclasses import dataclass


@dataclass
class FasterWhisperConf:
    """
    Configuration for the FasterWhisper ASR model.

    :param model_size: Whisper model size (e.g. "tiny", "base", "small", "medium", "large-v3", "turbo")
    :param device: Device to run on ("cpu" or "cuda")
    :param compute_type: Quantization type ("int8", "float16", "float32")
    :param language: Language code for transcription (e.g. "en"), or None for auto-detect
    :param beam_size: Beam size for decoding (1 = greedy, faster; 5 = default, more accurate)
    :param condition_on_previous_text: Whether to condition on previous text (False is faster for independent segments)
    """

    model_size: str = "base"
    device: str = "cpu"
    compute_type: str = "int8"
    language: str = "en"
    beam_size: int = 1
    condition_on_previous_text: bool = False
