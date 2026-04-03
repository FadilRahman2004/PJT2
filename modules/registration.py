"""
Module 11 — Unknown Person Registration
Captures multiple face frames, generates embeddings, and registers a new
person in the face database.
"""

import time
import numpy as np


class RegistrationManager:
    """
    Handles interactive face registration for unknown persons.

    Workflow:
        1. Track how many frames an unknown face persists.
        2. After the persistence threshold, prompt the user.
        3. Capture N face frames and compute embeddings.
        4. Store under a user-provided name.
    """

    def __init__(self, face_recognizer, persist_frames=30,
                 capture_count=10):
        """
        Parameters
        ----------
        face_recognizer : FaceRecognizer
            The recognizer instance (used for embedding + DB storage).
        persist_frames : int
            Frames an unknown must persist before prompting.
        capture_count : int
            How many face embeddings to capture during registration.
        """
        self.recognizer = face_recognizer
        self.persist_frames = persist_frames
        self.capture_count = capture_count

        self._unknown_counter = 0
        self._prompted = False
        self._registering = False

    # ── public API ────────────────────────────

    def tick_unknown(self):
        """
        Call once per frame when an unknown person is detected.
        Returns True when the system is ready to prompt.
        """
        if self._registering or self._prompted:
            return False
        self._unknown_counter += 1
        if self._unknown_counter >= self.persist_frames:
            return True
        return False

    def reset(self):
        """Reset counters (call when no unknown is visible)."""
        self._unknown_counter = 0
        self._prompted = False

    def start_registration(self):
        """Mark registration as in-progress."""
        self._registering = True
        self._prompted = True

    def finish_registration(self, name, embeddings):
        """
        Complete registration: store embeddings under *name*.

        Parameters
        ----------
        name : str
        embeddings : list[np.ndarray]
        """
        if embeddings:
            self.recognizer.register(name, embeddings)
            print(f"[Registration] '{name}' saved with "
                  f"{len(embeddings)} embeddings.")
        self._registering = False
        self._unknown_counter = 0

    @property
    def is_registering(self):
        return self._registering

    def collect_embedding(self, face_image):
        """
        Extract a single embedding for registration.

        Returns the embedding ndarray, or None on failure.
        """
        return self.recognizer.get_embedding(face_image)
