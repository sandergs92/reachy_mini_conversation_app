"""
TTS worker that runs Piper synthesis + Claptrap FX in a separate process
to avoid GIL contention, and pushes rendered audio to a Reachy speaker
via a lightweight pusher thread in the main process.

Optionally feeds rendered audio to a HeadWobbler for speech-driven head sway.
"""

import base64
import multiprocessing as mp
import os
import signal
import threading
import time

import numpy as np

_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_VOICE_PATH = os.path.join(_MODEL_DIR, "nl_NL-ronnie-medium.onnx")


def _tts_synth_process(
    text_queue: mp.Queue,
    audio_queue: mp.Queue,
    voice_path: str,
    output_sr: int,
):
    """
    Runs in a child process.  Reads text from *text_queue*, synthesizes
    with Piper + Claptrap FX, resamples to *output_sr*, and puts the
    resulting float32 numpy arrays onto *audio_queue*.

    A ``None`` sentinel on *audio_queue* marks end-of-utterance.
    A ``None`` on *text_queue* signals shutdown.
    """
    # Ignore SIGINT in the child — the parent handles Ctrl-C and
    # sends us a None sentinel (or terminates us) to shut down.
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    from pedalboard import (
        Bitcrush,
        Chorus,
        Clipping,
        Compressor,
        Gain,
        HighpassFilter,
        LowpassFilter,
        Pedalboard,
        PitchShift,
    )
    from piper import PiperVoice
    from scipy.signal import resample as sp_resample

    voice = PiperVoice.load(voice_path)
    board = Pedalboard(
        [
            PitchShift(semitones=4),
            HighpassFilter(cutoff_frequency_hz=800),
            LowpassFilter(cutoff_frequency_hz=6000),
            Bitcrush(bit_depth=8),
            Chorus(rate_hz=3.0, depth=0.15, mix=0.3),
            Gain(gain_db=10),
            Clipping(threshold_db=-3.0),
            Compressor(
                threshold_db=-15, ratio=6.0, attack_ms=5.0, release_ms=50.0,
            ),
            Gain(gain_db=3),
        ]
    )

    while True:
        try:
            text = text_queue.get(timeout=0.2)
        except Exception:
            continue

        if text is None:
            break

        try:
            for chunk in voice.synthesize(text):
                tts_sr = chunk.sample_rate

                # int16 bytes → float32
                audio_int16 = np.frombuffer(
                    chunk.audio_int16_bytes, dtype=np.int16,
                )
                audio_float = audio_int16.astype(np.float32) / 32768.0

                # Pedalboard expects (channels, samples)
                audio_float = audio_float.reshape(1, -1)
                effected = board(audio_float, tts_sr)
                effected = effected.flatten()

                # Resample Piper SR → Reachy output SR
                if tts_sr != output_sr:
                    n_out = int(len(effected) * output_sr / tts_sr)
                    effected = sp_resample(effected, n_out)

                audio_queue.put((effected, output_sr))

        except Exception as e:
            print(f"[TTS process] Error: {e}")

        # End-of-utterance sentinel
        audio_queue.put(None)


class TTSWorker:
    """
    Manages a child *process* for Piper synthesis (avoids GIL contention
    with the main-loop ONNX inference) and a lightweight *thread* that
    pulls rendered audio and pushes it to Reachy's speaker.

    Optionally feeds rendered audio chunks to a HeadWobbler for
    speech-driven head sway.

    Usage::

        tts = TTSWorker(mini.media, output_sr, head_wobbler=wobbler)
        tts.start()
        tts.say("Hallo wereld")   # non-blocking
        ...
        tts.stop()
    """

    def __init__(
        self,
        media,
        output_sr: int,
        voice_path: str = DEFAULT_VOICE_PATH,
        head_wobbler=None,
    ):
        self._media = media
        self._output_sr = output_sr
        self._voice_path = voice_path
        self._head_wobbler = head_wobbler

        # mp.Queue for text → child process, and audio ← child process
        self._text_q: mp.Queue = mp.Queue()
        self._audio_q: mp.Queue = mp.Queue()

        self._process = mp.Process(
            target=_tts_synth_process,
            args=(self._text_q, self._audio_q, self._voice_path, output_sr),
            daemon=True,
        )
        # Lightweight pusher thread in main process
        self._pusher_stop = threading.Event()
        self._pusher = threading.Thread(target=self._push_loop, daemon=True)

    def start(self):
        self._process.start()
        self._pusher.start()

    def stop(self):
        # Signal the child process to exit gracefully
        try:
            self._text_q.put_nowait(None)
        except Exception:
            pass
        self._pusher_stop.set()
        self._process.join(timeout=3.0)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=1.0)
        self._pusher.join(timeout=2.0)

    def say(self, text: str):
        """Enqueue text for synthesis (non-blocking)."""
        # Reset wobbler state for the new utterance so timing is fresh
        if self._head_wobbler is not None:
            self._head_wobbler.reset()
        self._text_q.put(text)

    def _push_loop(self):
        """
        Runs in the main process.  Pulls rendered float32 audio from the
        child process, pushes it to the Reachy speaker, and optionally
        feeds it to the HeadWobbler for speech-driven head sway.
        """
        while not self._pusher_stop.is_set():
            try:
                item = self._audio_q.get(timeout=0.1)
            except Exception:
                continue

            if item is None:
                # End-of-utterance sentinel
                if self._head_wobbler is not None:
                    self._head_wobbler.reset()
                continue

            samples, sr = item

            # Feed HeadWobbler (expects base64-encoded int16)
            if self._head_wobbler is not None:
                int16_samples = np.clip(
                    samples * 32768.0, -32768, 32767,
                ).astype(np.int16)
                b64 = base64.b64encode(int16_samples.tobytes()).decode("ascii")
                self._head_wobbler.feed(b64)

            # Push to Reachy speaker
            self._media.push_audio_sample(samples)

            # Pace playback so we don't flood the buffer
            time.sleep(len(samples) / self._output_sr)
