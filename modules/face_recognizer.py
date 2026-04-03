"""
Module 4 — Face Recognition (Known vs Unknown)
Generates 512-d embeddings via FaceNet (facenet-pytorch).
Compares against a local JSON face database using cosine similarity.
Self-learning: updates embeddings on each successful match.
"""

import json
import os
import threading
import time

import cv2
import numpy as np
import torch
from PIL import Image
from facenet_pytorch import InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity
import config


class FaceRecognizer:
    """
    FaceNet-based face recognizer with JSON database backend.
    """

    def __init__(self, pretrained="vggface2", db_path="data/faces_db.json",
                 threshold=0.70, max_embeddings=50, enroll_threshold=25):
        """
        Parameters
        ----------
        pretrained : str
            Pretrained dataset — 'vggface2' or 'casia-webface'.
        db_path : str
            Path to the JSON face database.
        threshold : float
            Cosine similarity threshold for a positive match.
        max_embeddings : int
            Maximum embeddings stored per person.
        enroll_threshold : int
            Number of frames to track an unknown before auto-enrollment.
        """
        self.threshold = threshold
        self.max_embeddings = max_embeddings
        self.db_path = db_path
        self._lock = threading.Lock()
        
        # Self-Learning State
        self.candidates = {} # {track_id: [embeddings]}
        self.enroll_threshold = enroll_threshold
        self.visitor_count = 0 

        # Device selection
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Initialize FaceNet
        self.model = InceptionResnetV1(
            pretrained=pretrained
        ).eval().to(self.device)

        self.database = self._load_db()
        self.identity_cache = {} # {track_id: {"name": str, "conf": float, "expiry": int}}
        
        # Continuous Dataset Tracking
        self.dataset_counters = {} # {name: count}
        os.makedirs(config.FACE_DATASET_DIR, exist_ok=True)
        
        print(f"[FaceRecognizer] Using device: {self.device}")
        print(f"[FaceRecognizer] Loaded {len(self.database)} known faces")

    # ── public API ────────────────────────────

    def recognize(self, face_image, person_id=None):
        """
        Identify a person from a face image.

        Parameters
        ----------
        face_image : np.ndarray
            RGB face image (160×160 recommended, aligned).
        person_id : int | None
            Optional track ID for identity caching.

        Returns
        -------
        dict with keys:
            name       : str — person name or "Unknown"
            confidence : float — best cosine similarity (0 if unknown)
            embedding  : np.ndarray — 512-d vector for this face
        """
        # 0. Check Cache for performance optimization (Return cached identity even if face_image is None)
        if person_id is not None and person_id in self.identity_cache:
            cache = self.identity_cache[person_id]
            if cache["expiry"] > 0 or face_image is None:
                if cache["expiry"] > 0: cache["expiry"] -= 1
                return {
                    "name": cache["name"],
                    "confidence": cache["conf"],
                    "embedding": None,
                }

        if face_image is None:
            return {"name": "Unknown", "confidence": 0.0, "embedding": None}

        embedding = self._get_embedding(face_image)
        if embedding is None:
            return {"name": "Unknown", "confidence": 0.0, "embedding": None}

        best_name = "Unknown"
        best_score = 0.0

        with self._lock:
            for name, data in self.database.items():
                avg_emb = np.array(data["average_embedding"])
                score = cosine_similarity(
                    embedding.reshape(1, -1),
                    avg_emb.reshape(1, -1)
                )[0][0]
                if score > best_score:
                    best_score = score
                    best_name = name

        if best_score < self.threshold:
            return {
                "name": "Unknown",
                "confidence": float(best_score),
                "embedding": embedding,
            }

        # Self-learning: update embeddings with diversity check
        self._update_embedding(best_name, embedding)
        
        # Update Cache if high confidence
        if person_id is not None and best_score >= config.FACE_CACHE_THRESHOLD:
            self.identity_cache[person_id] = {
                "name": best_name,
                "conf": float(best_score),
                "expiry": config.FACE_CACHE_EXPIRY
            }

        # --- Continuous Learning: Dataset Capture ---
        if person_id is not None and best_score >= self.threshold:
            self.dataset_counters[best_name] = self.dataset_counters.get(best_name, 0) + 1
            if self.dataset_counters[best_name] % config.FACE_CAPTURE_INTERVAL == 0:
                self._save_dataset_image(best_name, face_image)

        return {
            "name": best_name,
            "confidence": float(best_score),
            "embedding": embedding,
        }

    def auto_enroll(self, track_id, embedding):
        """
        Silently track and potentially enroll a new person.
        """
        if track_id is None:
            return None

        with self._lock:
            if track_id not in self.candidates:
                self.candidates[track_id] = []
            
            self.candidates[track_id].append(embedding)
            
            # If reached threshold, enroll as Visitor
            if len(self.candidates[track_id]) >= self.enroll_threshold:
                # Find current visitor index
                existing_visitors = [n for n in self.database.keys() if n.startswith("Visitor_")]
                idx = len(existing_visitors) + 1
                name = f"Visitor_{idx}"
                
                embeddings = self.candidates[track_id]
                avg = np.mean(embeddings, axis=0).tolist()
                
                self.database[name] = {
                    "embeddings": [e.tolist() for e in embeddings],
                    "average_embedding": avg,
                    "last_seen": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                self._save_db()
                del self.candidates[track_id] # Clear candidate
                return name
                
        return None

    def register(self, name, embeddings_list):
        """
        Register a new person with a list of embeddings.

        Parameters
        ----------
        name : str
            Person's name.
        embeddings_list : list[np.ndarray]
            List of 512-d face embeddings.
        """
        avg = np.mean(embeddings_list, axis=0).tolist()
        with self._lock:
            self.database[name] = {
                "embeddings": [e.tolist() for e in embeddings_list],
                "average_embedding": avg,
                "last_seen": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            self._save_db()

    def get_embedding(self, face_image):
        """Public wrapper for embedding extraction."""
        return self._get_embedding(face_image)

    def known_names(self):
        """Return list of all registered names."""
        with self._lock:
            return list(self.database.keys())

    # ── internals ─────────────────────────────

    def _get_embedding(self, face_image):
        """
        Extract 512-d embedding from face image.

        Parameters
        ----------
        face_image : np.ndarray
            RGB image, ideally 160×160.

        Returns
        -------
        np.ndarray or None
        """
        try:
            # Ensure 160×160
            if face_image.shape[:2] != (160, 160):
                face_image = cv2.resize(face_image, (160, 160))

            # Convert to float tensor [0,1] → normalize
            img = face_image.astype(np.float32) / 255.0
            # Normalize as FaceNet expects: (x - 127.5) / 128.0
            # but since we already [0,1], use (x - 0.5) / 0.5
            img = (img - 0.5) / 0.5

            # HWC → CHW → batch
            tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
            tensor = tensor.to(self.device)

            with torch.no_grad():
                embedding = self.model(tensor)

            return embedding.cpu().numpy().flatten()

        except Exception as e:
            print(f"[FaceRecognizer] Embedding error: {e}")
            return None

    def _update_embedding(self, name, new_embedding):
        """
        Diversity-Aware update. 
        Instead of just replacing the oldest, we keep embeddings that 
        are "different" (diverse angles/lighting) from the current average.
        """
        with self._lock:
            data = self.database[name]
            embs_list = [np.array(e) for e in data["embeddings"]]
            avg_emb = np.array(data["average_embedding"])
            
            # Similarity to current average
            sim_to_avg = cosine_similarity(
                new_embedding.reshape(1, -1),
                avg_emb.reshape(1, -1)
            )[0][0]
            
            # If the database isn't full yet, just add it
            if len(embs_list) < self.max_embeddings:
                data["embeddings"].append(new_embedding.tolist())
            else:
                # Database is full. Find the one most redundant to the average to replace.
                # OR, if the new one is VERY similar to the average, maybe skip it 
                # to keep more diverse samples (side views).
                
                if sim_to_avg > 0.98:
                    # Too redundant, don't waste a slot
                    return

                # Replace the embedding that is closest to the average (least unique)
                similarities = [
                    cosine_similarity(e.reshape(1, -1), avg_emb.reshape(1, -1))[0][0]
                    for e in embs_list
                ]
                redundant_idx = np.argmax(similarities)
                data["embeddings"][redundant_idx] = new_embedding.tolist()

            # Recompute average
            data["average_embedding"] = np.mean(
                [np.array(e) for e in data["embeddings"]], axis=0
            ).tolist()
            data["last_seen"] = time.strftime("%Y-%m-%d %H:%M:%S")
            self._save_db()

    def _save_dataset_image(self, name, face_image):
        """
        Save a face image to the person's dataset folder.
        Implements a circular buffer logic (max 50).
        """
        person_dir = os.path.join(config.FACE_DATASET_DIR, name)
        os.makedirs(person_dir, exist_ok=True)

        files = sorted([f for f in os.listdir(person_dir) if f.endswith(".jpg")])
        if len(files) >= config.FACE_DATASET_LIMIT:
            # Delete the oldest image (first in sorted list)
            try:
                os.remove(os.path.join(person_dir, files[0]))
            except Exception as e:
                print(f"[FaceRecognizer] Dataset cleanup error: {e}")

        # Save new image
        timestamp = int(time.time() * 1000)
        filename = f"{name}_{timestamp}.jpg"
        save_path = os.path.join(person_dir, filename)
        
        try:
            # Convert to BGR if needed for cv2.imwrite
            # (Note: face_image is usually RGB here as per docstring)
            if face_image is not None:
                bgr = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(save_path, bgr)
        except Exception as e:
            print(f"[FaceRecognizer] Dataset save error: {e}")

    def _load_db(self):
        """Load the face database from JSON."""
        if os.path.isfile(self.db_path):
            with open(self.db_path, "r") as f:
                return json.load(f)
        return {}

    def _save_db(self):
        """Persist the face database to JSON."""
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        with open(self.db_path, "w") as f:
            json.dump(self.database, f, indent=2)
