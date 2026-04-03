"""
Module 1 — Frame Capture
Continuously captures video frames from the webcam in a dedicated thread.
"""

import cv2
import threading
import queue
import time


class FrameCapture:
    """Threaded webcam frame capture with buffered queue output."""

    def __init__(self, camera_index=0, width=640, height=480,
                 target_fps=25, queue_size=4):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps

        self._cap = None
        self._queue = queue.Queue(maxsize=queue_size)
        self._running = False
        self._thread = None
        self._lock = threading.Lock()

        # FPS tracking
        self._fps = 0.0
        self._frame_count = 0
        self._fps_start = time.time()

    # ── public API ────────────────────────────

    def start(self):
        """Open the camera and begin capture in a background thread."""
        self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera index {self.camera_index}"
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop,
                                        daemon=True)
        self._thread.start()
        return self

    def stop(self):
        """Stop the capture thread and release the camera."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        if self._cap is not None:
            self._cap.release()

    def read(self, timeout=1.0):
        """
        Return the next frame from the buffer.
        Returns None if no frame is available within *timeout* seconds.
        """
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    @property
    def fps(self):
        """Current measured capture FPS."""
        return self._fps

    # ── internals ─────────────────────────────

    def _capture_loop(self):
        while self._running:
            loop_start = time.time()
            ret, frame = self._cap.read()
            if not ret:
                continue

            # Resize for consistent downstream processing
            frame = cv2.resize(frame, (self.width, self.height))

            # Drop old frames if the consumer is slow
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
            self._queue.put(frame)

            # FPS bookkeeping
            self._frame_count += 1
            elapsed = time.time() - self._fps_start
            if elapsed >= 1.0:
                self._fps = self._frame_count / elapsed
                self._frame_count = 0
                self._fps_start = time.time()

            # Throttle to target FPS
            sleep_time = self.frame_interval - (time.time() - loop_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

    # ── context-manager support ───────────────

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
