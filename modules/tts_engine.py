"""
Module 10 — TTS Engine
Offline text-to-speech narration via pyttsx3.
Runs speech in a background thread with cooldown to prevent overlap.
"""

import threading
import time
import queue

import pyttsx3


class TTSEngine:
    """
    Non-blocking text-to-speech engine.

    Narration requests are queued and spoken sequentially in a
    background thread. A cooldown prevents the same message from
    being repeated too quickly.
    """

    def __init__(self, rate=175, volume=0.9, cooldown=5.0):
        """
        Parameters
        ----------
        rate : int
            Speech words-per-minute.
        volume : float
            Volume level (0.0 – 1.0).
        cooldown : float
            Minimum seconds between identical narration strings.
        """
        self.rate = rate
        self.volume = volume
        self.cooldown = cooldown

        self._queue = queue.Queue()
        self._running = False
        self._thread = None
        self._last_spoken = {}  # text → timestamp

    # ── public API ────────────────────────────

    def start(self):
        """Start the background TTS thread."""
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        """Stop the TTS thread."""
        self._running = False
        if self._thread is not None:
            self._queue.put(None)  # sentinel
            self._thread.join(timeout=5.0)

    def speak(self, text):
        """
        Queue *text* for narration.

        Duplicate text within the cooldown window is silently dropped.
        """
        now = time.time()
        last = self._last_spoken.get(text, 0)
        if now - last < self.cooldown:
            return
        self._last_spoken[text] = now
        self._queue.put(text)

    # ── internals ─────────────────────────────

    def _run_loop(self):
        # pyttsx3 engine must be created in the same thread that uses it
        engine = pyttsx3.init()
        engine.setProperty("rate", self.rate)
        engine.setProperty("volume", self.volume)

        while self._running:
            try:
                text = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if text is None:  # stop sentinel
                break

            try:
                engine.say(text)
                engine.runAndWait()
            except Exception:
                pass  # swallow TTS errors to keep thread alive

        engine.stop()

    # ── context-manager support ───────────────

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
