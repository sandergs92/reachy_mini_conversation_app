import shutil
import sys

# ANSI codes
SPEAKER_COLORS = ["\033[94m", "\033[92m", "\033[93m", "\033[95m"]
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CLEAR_LINE = "\033[2K"
MOVE_UP = "\033[1A"
RESET_FG = "\x1b[39m"
RESET_BOLD = "\x1b[22m"
UNDERLINE = "\x1b[4m"
NO_UNDERLINE = "\x1b[24m"


def fmt_time(ts: float) -> str:
    minutes = int(ts // 60)
    seconds = int(ts % 60)
    ms = int((ts % 1) * 1000)
    return f"{minutes:02d}:{seconds:02d}.{ms:03d}"


def format_segment(
    speaker_id: int,
    timestamp: float,
    end_timestamp: float,
    text: str,
    active: bool = False,
) -> str:
    color = SPEAKER_COLORS[speaker_id % len(SPEAKER_COLORS)]
    cursor = "\u258c" if active else ""
    return (
        f"{DIM}[{fmt_time(timestamp)} - {fmt_time(end_timestamp)}]{RESET_BOLD}{RESET_FG} "
        f"{color}{BOLD}Speaker {speaker_id + 1}:{RESET_BOLD}{RESET_FG} "
        f"{text}{cursor}"
    )


def _visible_len(text: str) -> int:
    """Return the number of visible (non-ANSI-escape) characters in *text*."""
    length = 0
    i = 0
    while i < len(text):
        if text[i] == "\033":
            while i < len(text) and text[i] != "m":
                i += 1
        else:
            length += 1
        i += 1
    return length


def format_speaker(spk: int | None) -> tuple[str, str]:
    if spk is None:
        # label, word_style
        return (f"{DIM}[?]{RESET_BOLD}{RESET_FG}", f"{DIM}{UNDERLINE}")

    color = SPEAKER_COLORS[spk % len(SPEAKER_COLORS)]
    label = f"{color}{BOLD}[S{spk + 1}]{RESET_BOLD}{RESET_FG}"
    word_style = f"{color}{UNDERLINE}"
    return label, word_style


def tag_words(
    words, speaker_lookup, *, underline_words: bool = True, dim_words: bool = False
) -> str:
    if not words:
        return ""

    tagged_parts = []
    current_spk = None
    current_words = []

    def fmt_label_and_color(spk: int | None):
        if spk is None:
            return (f"{DIM}[?]{RESET_BOLD}{RESET_FG}", f"{DIM}")
        color = SPEAKER_COLORS[spk % len(SPEAKER_COLORS)]
        label = f"{color}{BOLD}[S{spk + 1}]{RESET_BOLD}{RESET_FG}"
        return label, color

    def flush(spk, ws):
        if not ws:
            return
        label, color = fmt_label_and_color(spk)
        text = " ".join(ws)

        word_style = ""
        word_end = ""
        if dim_words:
            word_style += DIM
        # keep same speaker color for words
        if spk is not None:
            word_style += color
        if underline_words:
            word_style += UNDERLINE
            word_end = NO_UNDERLINE  # close underline if we opened it

        tagged_parts.append(f"{label} {word_style}{text}{word_end}{RESET_FG}")

    for w in words:
        spk = speaker_lookup.get_word_speaker(w)
        if spk != current_spk and current_words:
            flush(current_spk, current_words)
            current_words = []
        current_spk = spk
        current_words.append(w[2])

    flush(current_spk, current_words)
    return " ".join(tagged_parts)


class StreamingDisplay:
    """Handles streaming display with multi-line support."""

    def __init__(self):
        self.has_active = False
        self.last_line_count = 0
        self._width = shutil.get_terminal_size().columns

    def _count_lines(self, text: str) -> int:
        """Count how many terminal rows *text* occupies, including wrapping."""
        total = 0
        for line in text.split("\n"):
            vis = _visible_len(line)
            # An empty line still takes one row
            total += max(1, (vis + self._width - 1) // self._width)
        return total

    def clear_active(self):
        if self.has_active and self.last_line_count > 0:
            sys.stdout.write(f"\r{CLEAR_LINE}")
            for _ in range(self.last_line_count - 1):
                sys.stdout.write(f"{MOVE_UP}{CLEAR_LINE}")
            sys.stdout.write("\r")
            sys.stdout.flush()
        self.has_active = False
        self.last_line_count = 0

    def print_final(self, text: str):
        self.clear_active()
        print(text)

    def update_active(self, text: str):
        self.clear_active()
        sys.stdout.write(text)
        sys.stdout.flush()
        self.last_line_count = self._count_lines(text)
        self.has_active = True
