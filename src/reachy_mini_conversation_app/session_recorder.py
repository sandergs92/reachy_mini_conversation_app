import queue
import wave
import logging
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray


logger = logging.getLogger(__name__)

_AUDIO_SAMPLE_RATE = 24000
_AUDIO_CHANNELS = 1
_AUDIO_SAMPWIDTH = 2  # int16


def _resolve_run_dir(base: Path) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    index = 0
    while (base / str(index)).exists():
        index += 1
    run_dir = base / str(index)
    run_dir.mkdir()
    return run_dir


class SessionRecorder:
    def __init__(self, base_dir: str = "data") -> None:
        self._run_dir = _resolve_run_dir(Path(base_dir))
        self._images_dir = self._run_dir / "images"
        self._audio_dir = self._run_dir / "audio"
        self._log_path = self._run_dir / "log" / "log.txt"

        self._images_dir.mkdir()
        self._audio_dir.mkdir()
        self._log_path.parent.mkdir()

        self._queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)

        self._file_handler: logging.FileHandler | None = None

    def start(self) -> None:
        self._file_handler = logging.FileHandler(self._log_path, encoding="utf-8")
        self._file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        logging.getLogger().addHandler(self._file_handler)
        self._thread.start()
        logger.info("SessionRecorder started, run dir: %s", self._run_dir)

    def stop(self) -> None:
        self._stop_event.set()
        self._queue.put(None)
        self._thread.join()
        if self._file_handler is not None:
            logging.getLogger().removeHandler(self._file_handler)
            self._file_handler.close()

    def record_frame(self, frame: NDArray[np.uint8]) -> None:
        self._queue.put(("image", time.time(), frame))

    def record_audio(self, chunk: bytes) -> None:
        self._queue.put(("audio", time.time(), chunk))

    def _worker(self) -> None:
        while not self._stop_event.is_set() or not self._queue.empty():
            try:
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if item is None:
                break

            kind, ts, data = item
            filename = f"{ts:.6f}"

            try:
                if kind == "image":
                    cv2.imwrite(str(self._images_dir / f"{filename}.png"), data)
                elif kind == "audio":
                    path = str(self._audio_dir / f"{filename}.wav")
                    with wave.open(path, "wb") as wf:
                        wf.setnchannels(_AUDIO_CHANNELS)
                        wf.setsampwidth(_AUDIO_SAMPWIDTH)
                        wf.setframerate(_AUDIO_SAMPLE_RATE)
                        wf.writeframes(data)
            except Exception as e:
                logger.error("SessionRecorder write error (%s): %s", kind, e)
