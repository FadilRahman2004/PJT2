"""
Module 5 — Emotion Detection (Hybrid: HuggingFace ViT + Geometric Face Mesh)
Uses 'trpham/face-expression-recognition' (ViT trained on AffectNet-7) 
as primary, with a Face Mesh geometric layer as a secondary validator.

Emotion labels: Angry, Contempt, Disgust, Fear, Happy, Neutral, Sad, Surprise
"""

import os
import cv2
import time
import numpy as np
import torch
import mediapipe as mp
import config

# ─── Constants ─────────────────────────────────────────────────────────────────
VALENCE_MAP = {
    "Angry": "negative",    "Contempt": "negative",
    "Disgust": "negative",  "Fear": "negative",
    "Happy": "positive",    "Surprise": "positive",
    "Sad": "negative",      "Neutral": "neutral",
}

# ─── Primary Engine ─────────────────────────────────────────────────────────────
class HFEmotionEngine:
    """
    Primary engine: A ViT/EfficientNet-based model from HuggingFace.
    Falls back to hsemotion ensemble if HF model cannot be loaded.
    """
    
    # Best model: trained on AffectNet-7, robust for real-world webcam use
    HF_MODEL_ID = "dima806/facial_emotions_image_detection"
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.pipeline = None
        self._try_load_hf_model()
        
        # Fallback: hsemotion ensemble
        self.fallback_experts = []
        if self.pipeline is None:
            print("[EmotionDetector] HF model unavailable. Loading hsemotion fallback...")
            self._load_hsemotion_fallback()

    def _try_load_hf_model(self):
        try:
            from transformers import pipeline as hf_pipeline
            print(f"[EmotionDetector] Loading HuggingFace model: {self.HF_MODEL_ID}")
            self.pipeline = hf_pipeline(
                task="image-classification",
                model=self.HF_MODEL_ID,
                device=0 if self.device == "cuda" else -1,
                top_k=7  # Return all 7 emotion classes
            )
            print(f"[EmotionDetector] HuggingFace model ready.")
        except Exception as e:
            print(f"[EmotionDetector] HF model load failed: {e}")
            self.pipeline = None

    def _load_hsemotion_fallback(self):
        try:
            from hsemotion.facial_emotions import HSEmotionRecognizer
            orig_load = torch.load
            torch.load = lambda *args, **kwargs: orig_load(*args, weights_only=False, **kwargs)
            try:
                experts = ['enet_b0_8_best_vgaf', 'enet_b0_8_best_afew', 'enet_b0_8_va_mtl']
                weights = [0.50, 0.30, 0.20]
                for name, w in zip(experts, weights):
                    m = HSEmotionRecognizer(model_name=name, device=self.device)
                    self.fallback_experts.append((m, w))
            finally:
                torch.load = orig_load
            print(f"[EmotionDetector] {len(self.fallback_experts)} hsemotion fallback experts ready.")
        except Exception as e:
            print(f"[EmotionDetector] hsemotion fallback failed: {e}")

    def predict(self, face_image_bgr: np.ndarray) -> np.ndarray:
        """
        Returns a probability vector over 7 canonical classes:
        [Angry, Contempt, Disgust, Fear, Happy, Neutral, Sad, Surprise]
        """
        label_order = ["Angry", "Contempt", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
        scores = np.full(8, 1e-4)  # Start with tiny non-zero priors
        
        if self.pipeline is not None:
            return self._predict_hf(face_image_bgr, label_order)
        elif self.fallback_experts:
            return self._predict_hsemotion(face_image_bgr, label_order)
        
        return scores

    def _predict_hf(self, face_bgr: np.ndarray, label_order: list) -> np.ndarray:
        """Run HuggingFace model and return probability vector."""
        from PIL import Image
        scores = np.full(8, 1e-4)
        try:
            face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(face_rgb)
            results = self.pipeline(pil_image)
            for r in results:
                label = r["label"]
                # Normalize label names
                label = label.capitalize()
                if label in label_order:
                    idx = label_order.index(label)
                    scores[idx] = r["score"]
                elif label.lower() == "happy":
                    scores[label_order.index("Happy")] = r["score"]
                elif label.lower() == "angry":
                    scores[label_order.index("Angry")] = r["score"]
        except Exception as e:
            print(f"[HFEmotionEngine] Predict error: {e}")
        
        # Protect against all-zero
        total = np.sum(scores)
        if total < 0.01:
            scores[5] = 1.0  # Default to Neutral
        return scores / np.sum(scores)

    def _predict_hsemotion(self, face_bgr: np.ndarray, label_order: list) -> np.ndarray:
        """Run hsemotion ensemble and return probability vector."""
        # HSEmotion labels: 0:Angry 1:Contempt 2:Disgust 3:Fear 4:Happy 5:Neutral 6:Sad 7:Surprise
        combined = np.zeros(8)
        total_w = 0.0
        for expert, w in self.fallback_experts:
            try:
                _, raw = expert.predict_emotions(face_bgr, logits=False)
                arr = np.array(raw[:8]) if len(raw) >= 8 else np.pad(raw, (0, 8-len(raw)))
                combined += arr * w
                total_w += w
            except Exception:
                continue
        if total_w == 0:
            combined[5] = 1.0  # Neutral default
        else:
            combined /= total_w
        return combined / np.sum(combined)


# ─── Main Detector ──────────────────────────────────────────────────────────────
class EmotionDetector:
    """
    Hybrid Emotion Detector:
    1. Primary: HuggingFace ViT model (AffectNet-7 trained)
    2. Geometric validator: MediaPipe Face Mesh (468 landmarks)
    3. Temporal smoother: EMA per person
    """

    LABEL_MAP     = ["Angry", "Contempt", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
    VALENCE_MAP   = ["negative","negative","negative","negative","positive","neutral","negative","positive"]

    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Primary engine
        self.engine = HFEmotionEngine(device=self.device)

        # Geometric validator (Face Mesh)
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # Temporal EMA memory {person_id: {vector, last_seen}}
        self.emotion_histories = {}
        # Per-person frame counters for throttling
        self.frame_counters = {}
        
        print("[EmotionDetector] Hybrid detector ready (HF + Geometric + EMA).")

    def detect(self, face_image, person_id=None):
        """
        Main detection pipeline.
        face_image: BGR numpy array (aligned face crop) or None (to use cache)
        """
        # 0. Cache Check: Return last known emotion if skipping alignment
        if face_image is None:
            p_key = person_id if person_id is not None else 0
            if p_key in self.emotion_histories:
                return self._vec_to_result(self.emotion_histories[p_key]["vector"])
            return self._default()

        if face_image.size == 0:
            return self._default()

        h, w = face_image.shape[:2]
        if h < config.EMOTION_MIN_FACE_SIZE or w < config.EMOTION_MIN_FACE_SIZE:
            if person_id is not None and person_id in self.emotion_histories:
                return self._vec_to_result(self.emotion_histories[person_id]["vector"])
            return self._default()

        try:
            # throttling: Only run neural model every N frames per person
            skip = config.EMOTION_SKIP_FRAMES
            p_key = person_id if person_id is not None else 0
            self.frame_counters[p_key] = self.frame_counters.get(p_key, 0) + 1
            
            should_detect = (self.frame_counters[p_key] % skip == 0) or (p_key not in self.emotion_histories)
            
            if should_detect:
                # 1. Neural prediction
                neural_vec = self.engine.predict(face_image)
            else:
                # Use last vector if skipping
                neural_vec = self.emotion_histories[p_key]["vector"].copy()

            # 2. Geometric correction
            geo_mults = self._geometry_multipliers(face_image)
            for emo, mult in geo_mults.items():
                if emo in self.LABEL_MAP:
                    neural_vec[self.LABEL_MAP.index(emo)] *= mult
            neural_vec /= np.sum(neural_vec)

            # 3. Temporal EMA smoothing
            alpha = config.EMOTION_ALPHA
            if person_id is not None:
                if person_id in self.emotion_histories:
                    prev = self.emotion_histories[person_id]["vector"]
                    neural_vec = alpha * neural_vec + (1 - alpha) * prev
                self.emotion_histories[person_id] = {
                    "vector": neural_vec,
                    "last_seen": time.time()
                }
                if len(self.emotion_histories) > 50:
                    self._cleanup()

            return self._vec_to_result(neural_vec)

        except Exception as e:
            print(f"[EmotionDetector] Error: {e}")
            return self._default()

    def _geometry_multipliers(self, face_bgr: np.ndarray) -> dict:
        """
        Analyse facial geometry to suppress/boost neural predictions.
        """
        mults = {"Happy": 1.0, "Angry": 1.0, "Surprise": 1.0, "Fear": 1.0, "Sad": 1.0}

        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        res = self.face_mesh.process(rgb)
        if not res.multi_face_landmarks:
            return mults

        lm = res.multi_face_landmarks[0].landmark

        def d(i, j):
            a, b = lm[i], lm[j]
            return np.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

        eye_dist = d(33, 263) + 1e-6  # Inter-pupil baseline

        # ── Smile Detection ─────────────────────────────────
        # Mouth width vs height
        mouth_w = d(61, 291)
        mouth_h = d(13, 14) + 1e-6
        smile_ratio = mouth_w / mouth_h

        if smile_ratio > 3.2:
            mults["Happy"] = 2.0
        elif smile_ratio > 2.4:
            mults["Happy"] = 1.5
        elif smile_ratio > 1.8:
            mults["Happy"] = 1.2

        # ── Brow Furrow → Anger ──────────────────────────────
        brow_dist = d(55, 285)
        brow_ratio = brow_dist / eye_dist
        if brow_ratio < 0.17:
            mults["Angry"] = 1.7
            mults["Neutral"] = 0.7
        elif brow_ratio < 0.22:
            mults["Angry"] = 1.2

        # ── Eyebrow Raise → Surprise / Fear Gate ────────────
        # Distance from inner brow to eye centre (vertical proxy)
        left_raise  = d(223, 33)   # Left brow to left eye
        right_raise = d(443, 263)  # Right brow to right eye
        avg_raise   = (left_raise + right_raise) / 2.0
        raise_ratio = avg_raise / eye_dist

        # Fear/Surprise require raised brows – if flat, suppress them hard
        if raise_ratio < 0.30:
            mults["Fear"]     = 0.30   # Strong suppression when brows are down
            mults["Surprise"] = 0.30
        elif raise_ratio > 0.60:
            mults["Surprise"] = 1.6
            mults["Fear"]     = 1.3

        # ── Drooping mouth corners → Sadness ──────────────────
        # Compare left/right mouth corners vs mouth centre height
        upper_lip = lm[13]
        left_mc   = lm[61]
        right_mc  = lm[291]
        avg_corner_y = (left_mc.y + right_mc.y) / 2.0
        if avg_corner_y > upper_lip.y + 0.01:  # Corners below upper lip
            mults["Sad"] = 1.4

        return mults

    def _vec_to_result(self, vec: np.ndarray) -> dict:
        idx = int(np.argmax(vec))
        return {
            "emotion":    self.LABEL_MAP[idx],
            "confidence": float(vec[idx]),
            "valence":    self.VALENCE_MAP[idx],
            "scores":     {lbl: float(v) for lbl, v in zip(self.LABEL_MAP, vec)},
        }

    def _cleanup(self):
        now = time.time()
        dead = [pid for pid, d in self.emotion_histories.items()
                if now - d["last_seen"] > 30]
        for pid in dead:
            del self.emotion_histories[pid]

    @staticmethod
    def _default():
        return {"emotion": "Neutral", "confidence": 0.0, "valence": "neutral", "scores": {}}
