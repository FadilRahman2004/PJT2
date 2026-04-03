"""
Module 13 — Speech-To-Text (STT) Engine
Provides "Ears" to the assistant. Runs a background thread listening
to the microphone, waits for a wake word, and transcribes the user's spoken request.
"""

import threading
import queue
import time
import speech_recognition as sr

class STTEngine:
    def __init__(self, use_stt=True, wake_word="assistant", timeout=5, phrase_limit=10, tts_engine=None):
        """
        Parameters
        ----------
        use_stt : bool
            Enable or disable listening.
        wake_word : str
            The word that triggers the system to start recording a query.
        timeout : int
            Seconds to listen for speech before giving up.
        phrase_limit : int
            Max duration of the spoken command.
        tts_engine : TTSEngine | None
            Used to give audio feedback (e.g., "I'm listening").
        """
        self.enabled = use_stt
        self.wake_word = wake_word.lower()
        self.timeout = timeout
        self.phrase_limit = phrase_limit
        self.tts = tts_engine
        
        self.recognizer = sr.Recognizer()
        self.mic = None
        self.query_queue = queue.Queue()
        
        self.is_running = False
        self._thread = None

    def start(self):
        """Starts the background listening thread."""
        if not self.enabled:
            print("[STTEngine] Disabled by config.")
            return

        try:
            self.mic = sr.Microphone()
            with self.mic as source:
                print("[STTEngine] Calibrating microphone for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1.0)
                
            self.is_running = True
            self._thread = threading.Thread(target=self._listen_loop, daemon=True)
            self._thread.start()
            print(f"[STTEngine] Ready. Wake word is '{self.wake_word}'.")
        except Exception as e:
            print(f"[STTEngine] Failed to start microphone: {e}")
            self.enabled = False

    def stop(self):
        """Signals the background thread to stop."""
        self.is_running = False
        if self._thread:
            self._thread.join(timeout=2)
            print("[STTEngine] Stopped.")

    def get_query(self):
        """
        Retrieves a transcribed query from the queue if one exists.
        Returns the string query, or None.
        """
        try:
            return self.query_queue.get_nowait()
        except queue.Empty:
            return None

    def _listen_loop(self):
        """The background loop that constantly listens for the wake word."""
        while self.is_running:
            with self.mic as source:
                # 1. Background listening (passive)
                try:
                    # Listen for quick bursts of audio (looking for wake word)
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=3)
                except sr.WaitTimeoutError:
                    continue
                except Exception as e:
                    print(f"[STTEngine] Passive listen error: {e}")
                    time.sleep(1)
                    continue

                # 2. Check if the audio contains the wake word
                try:
                    text = self.recognizer.recognize_google(audio).lower()
                    if self.wake_word in text:
                        print(f"[STTEngine] Wake word '{self.wake_word}' detected!")
                        
                        # Handle continuous speech (e.g., "assistant how are you")
                        parts = text.split(self.wake_word, 1)
                        query_tail = parts[1].strip() if len(parts) > 1 else ""
                        
                        if query_tail:
                            # The user said the question in the same breath
                            print(f"[STTEngine] Fast Query: '{query_tail}'")
                            self.query_queue.put(query_tail)
                        else:
                            # The user paused after saying the wake word
                            if self.tts:
                                self.tts.speak("I'm listening")
                            self._active_listen(source)
                except sr.UnknownValueError:
                    pass # Unintelligible background noise, ignore
                except sr.RequestError as e:
                    print(f"[STTEngine] API unavailable: {e}")

    def _active_listen(self, source):
        """Listens deeply for the user's actual question after the wake word is heard."""
        try:
            # We are already inside the 'with self.mic as source' block from the caller
            audio = self.recognizer.listen(source, timeout=self.timeout, phrase_time_limit=self.phrase_limit)
            query = self.recognizer.recognize_google(audio)
            print(f"[STTEngine] User asked: '{query}'")
            self.query_queue.put(query)
        except sr.WaitTimeoutError:
            print("[STTEngine] Active listen timed out. No question heard.")
            if self.tts:
               self.tts.speak("I didn't hear a question.")
        except sr.UnknownValueError:
            print("[STTEngine] Could not understand the question.")
        except Exception as e:
             print(f"[STTEngine] Active listen error: {e}")
