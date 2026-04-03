"""
Module 7 — Context Engine
Aggregates detections over time to reason about the room's atmosphere (vibe)
and social interactions between people.
"""

import time
from collections import deque


class ContextEngine:
    """
    Maintains situational awareness by analyzing trends in detections.
    """

    def __init__(self, vibe_window_fps=10, window_seconds=5.0):
        """
        Parameters
        ----------
        vibe_window_fps : int
            Approximate frequency of pipeline updates.
        window_seconds : float
            History length for calculating trends.
        """
        self.history_size = int(vibe_window_fps * window_seconds)
        # Deque of (timestamp, [person_results])
        self.history = deque(maxlen=self.history_size)
        
        # Current inferred state
        self.current_vibe = "Calm"
        self.known_count = 0
        self.unknown_count = 0

    def update(self, persons):
        """
        Update the engine with new detection results.
        
        Parameters
        ----------
        persons : list
            List of person dicts from the pipeline (identity, emotion, gesture).
        """
        self.history.append((time.time(), persons))
        
        # 1. Update counts
        self.known_count = sum(1 for p in persons if p["identity"]["name"] != "Unknown")
        self.unknown_count = len(persons) - self.known_count
        
        # 2. Update Room Vibe
        self.current_vibe = self._calculate_vibe()

    def _calculate_vibe(self):
        """
        Determines the 'Global Vibe Index' based on recent emotions and gestures.
        Uses temporal aggregation over the history window for stability.
        """
        if not self.history:
            return "Quiet"

        # Emotion Weights: High impact for strong emotions
        EMO_WEIGHTS = {
            "Angry": 3.0,
            "Disgust": 2.5,
            "Fear": 2.0,
            "Happy": 2.0,
            "Surprised": 1.5,
            "Sad": 1.5,
            "Neutral": 0.5
        }

        total_pos = 0.0
        total_neg = 0.0
        total_dynamic = 0.0
        person_ids = set()

        # 1. Aggregate over the entire history window
        for _, persons in self.history:
            for p in persons:
                person_ids.add(p["person_id"])
                emo_label = p.get("emotion", {}).get("emotion", "Neutral")
                gesture = p.get("gesture", "Standing")
                
                weight = EMO_WEIGHTS.get(emo_label, 1.0)
                
                if emo_label in ["Happy", "Surprised"]:
                    total_pos += weight
                elif emo_label in ["Angry", "Disgust", "Fear", "Sad"]:
                    total_neg += weight
                
                if gesture in ["Waving", "Pointing", "Raised Hand"]:
                    total_dynamic += 1.5
                elif gesture == "Handshake Offer":
                    total_dynamic += 2.0

        if not person_ids:
            return "Empty"

        # Normalize metrics per frame per person
        n_frames = len(self.history)
        n_people = len(person_ids)
        avg_pos = total_pos / (n_frames * n_people)
        avg_neg = total_neg / (n_frames * n_people)
        avg_dyn = total_dynamic / (n_frames * n_people)

        # 2. Vibe Decision Logic
        if avg_neg > 0.8: return "Conflict"
        if avg_neg > 0.4: return "Tense"
        if avg_pos > 0.6: return "Joyful"
        if avg_pos > 0.3: return "Positive"
        
        if avg_dyn > 0.5: return "Dynamic"
        if avg_dyn > 0.3: return "Social"
        
        # Check for Formal/Quiet
        latest_persons = self.history[-1][1]
        all_standing = all(p.get("gesture") in ["Standing", "Hands on Hips"] for p in latest_persons)
        if all_standing and avg_pos < 0.2 and avg_neg < 0.2:
            return "Formal"

        return "Calm"

    def get_summary(self):
        """
        Returns a high-level summary of the room state.
        """
        return {
            "vibe": self.current_vibe,
            "people_total": self.known_count + self.unknown_count,
            "known_count": self.known_count,
            "unknown_count": self.unknown_count,
        }

    def detect_interactions(self, persons):
        """
        Reason about social dynamics: Conversations, Presentations, etc.
        """
        interactions = []
        if len(persons) < 2:
            return interactions

        # 1. Detection of Conversations
        # People close to each other, likely facing
        for i, p1 in enumerate(persons):
            for p2 in persons[i+1:]:
                x1, _, x2, _ = p1["bbox"]
                cx1 = (x1 + x2) / 2
                ox1, _, ox2, _ = p2["bbox"]
                cx2 = (ox1 + ox2) / 2
                
                dist = abs(cx1 - cx2)
                if dist < 300: # Close proximity
                    name1 = p1["identity"]["name"]
                    name2 = p2["identity"]["name"]
                    emo1 = p1.get("emotion", {}).get("emotion", "Neutral")
                    emo2 = p2.get("emotion", {}).get("emotion", "Neutral")
                    
                    if emo1 in ["Happy", "Neutral", "Surprised"] and emo2 in ["Happy", "Neutral", "Surprised"]:
                        interactions.append(f"{name1} and {name2} appear to be in a conversation")
                    elif emo1 == "Angry" or emo2 == "Angry":
                        interactions.append(f"There is a tense confrontation between {name1} and {name2}")

        # 2. Detection of Presentations
        # One person gesturing/pointing while others are standing
        presenters = [p for p in persons if p.get("gesture") in ["Pointing", "Waving", "Raised Hand"]]
        listeners = [p for p in persons if p.get("gesture") in ["Standing", "Hands on Hips"]]
        
        if len(presenters) == 1 and len(listeners) >= 1:
            name = presenters[0]["identity"]["name"]
            interactions.append(f"{name} is leading a presentation or addressing the group")

        return interactions
