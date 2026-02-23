import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

WordTiming = Tuple[float, float, str]


class HypothesisBuffer:
    """Stabilizes streaming transcriptions by committing only the longest
    common prefix between consecutive hypothesis updates.

    Each word is a tuple of (start_time, end_time, text).
    """

    def __init__(self):
        self.commited_in_buffer: List[WordTiming] = []
        self.buffer: List[WordTiming] = []
        self.new: List[WordTiming] = []
        self.last_commited_time: float = 0
        self.last_commited_word: str = None

    def insert(self, new: List[WordTiming], offset: float = 0.0):
        """Insert a new hypothesis. Drops words that overlap with already
        committed content."""
        new = [(a + offset, b + offset, t) for a, b, t in new]
        self.new = [
            (a, b, t) for a, b, t in new if a > self.last_commited_time - 0.1
        ]

        if len(self.new) >= 1:
            a, b, t = self.new[0]
            if abs(a - self.last_commited_time) < 1:
                if self.commited_in_buffer:
                    cn = len(self.commited_in_buffer)
                    nn = len(self.new)
                    for i in range(1, min(min(cn, nn), 5) + 1):
                        c = " ".join(
                            [
                                self.commited_in_buffer[-j][2]
                                for j in range(1, i + 1)
                            ][::-1]
                        )
                        tail = " ".join(self.new[j - 1][2] for j in range(1, i + 1))
                        if c == tail:
                            for j in range(i):
                                removed = self.new.pop(0)
                                logger.debug(f"removing overlapping word: {removed}")
                            break

    def flush(self) -> List[WordTiming]:
        """Commit the longest common prefix between the previous and current
        hypothesis. Returns newly committed words."""
        commit = []
        while self.new:
            na, nb, nt = self.new[0]
            if len(self.buffer) == 0:
                break
            if nt == self.buffer[0][2]:
                commit.append((na, nb, nt))
                self.last_commited_word = nt
                self.last_commited_time = nb
                self.buffer.pop(0)
                self.new.pop(0)
            else:
                break
        self.buffer = self.new
        self.new = []
        self.commited_in_buffer.extend(commit)
        return commit

    def pop_commited(self, time: float):
        """Remove committed words older than the given time."""
        while self.commited_in_buffer and self.commited_in_buffer[0][1] <= time:
            self.commited_in_buffer.pop(0)

    def complete(self) -> List[WordTiming]:
        """Return the current uncommitted buffer (tentative/active words)."""
        return self.buffer
