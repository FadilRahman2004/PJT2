"""
Test Script: Social Scene Understanding (Room Vibe + Interactions + Narrative)
==========================================================================
Visualizes the high-level social data:
  - Global Room Vibe (Positive, Tense, Calm, etc.)
  - Social Interactions (Conversations, Presentations)
  - Heuristic Narrative summaries
"""

import cv2
import time
import numpy as np
import config
from modules.person_detector import PersonDetector
from modules.face_detector import FaceDetector
from modules.face_recognizer import FaceRecognizer
from modules.emotion_detector import EmotionDetector
from modules.gesture_detector import GestureDetector
from modules.context_engine import ContextEngine
from modules.narrative_engine import NarrativeEngine

def main():
    print("[Test] Initializing Full Social Scene Pipeline...")
    
    # 1. Init Detectors
    p_det = PersonDetector(confidence=config.YOLO_CONFIDENCE)
    f_det = FaceDetector()
    f_rec = FaceRecognizer(db_path=config.FACE_DB_PATH)
    e_det = EmotionDetector()
    g_det = GestureDetector()
    
    # 2. Init Scene Synthesis Engine
    ctx_engine = ContextEngine(vibe_window_fps=10, window_seconds=config.VIBE_WINDOW_SECONDS)
    nar_engine = NarrativeEngine(style="proactive", summary_cooldown=10.0) # Short cooldown for testing
    
    cap = cv2.VideoCapture(0)
    print("Controls: [Q] Quit")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # --- A. Detection Layer ---
        persons_detected = p_det.detect(frame)
        track_ids = [p["person_id"] for p in persons_detected]
        gestures = g_det.detect(frame, person_ids=track_ids)
        
        results = []
        for i, p in enumerate(persons_detected):
            tid = p["person_id"]
            bbox = p["bbox"]
            
            # Match Gesture
            gest = "Standing"
            kpts = None
            for g in gestures:
                if g["id"] == tid:
                    gest = g["gesture"]
                    kpts = g["keypoints"]
                    break
            
            # Face/Emotion
            body_crop = p["cropped_body"]
            aligned, _ = f_det.detect_and_align(body_crop)
            
            identity = {"name": "Unknown"}
            emotion  = {"emotion": "Neutral", "confidence": 0.0}
            
            if aligned is not None:
                identity = f_rec.recognize(aligned)
                emotion  = e_det.detect(aligned, person_id=tid)
                
            results.append({
                "person_id": tid,
                "bbox": bbox,
                "identity": identity,
                "emotion": emotion,
                "gesture": gest,
                "keypoints": kpts
            })

        # --- B. Synthesis Layer ---
        ctx_engine.update(results)
        summary = ctx_engine.get_summary()
        interactions = ctx_engine.detect_interactions(results)
        narrative = nar_engine.generate_narration(results, summary, interactions)
        
        if narrative:
            print(f"[Narrative] {narrative}")

        # --- C. Visual HUD ---
        # 1. Overlay for Vibe
        vibe = summary["vibe"]
        vibe_colors = {
            "Joyful": (0, 255, 0), "Positive": (100, 255, 100),
            "Conflict": (0, 0, 255), "Tense": (0, 100, 255),
            "Calm": (200, 200, 200), "Empty": (100, 100, 100)
        }
        col = vibe_colors.get(vibe, (255, 255, 255))
        
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 45), (30, 30, 30), -1)
        cv2.putText(frame, f"ROOM VIBE: {vibe.upper()}", (20, 30), 
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, col, 2)
        
        # 2. People Count
        count_str = f"People: {summary['people_total']} ({summary['known_count']} known)"
        cv2.putText(frame, count_str, (frame.shape[1]-300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 3. Interactions List
        for i, inter in enumerate(interactions):
            cv2.putText(frame, f"> {inter}", (20, 75 + i*25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # 4. Individual boxes
        for r in results:
            x1, y1, x2, y2 = r["bbox"]
            name = r["identity"]["name"]
            emo = r["emotion"]["emotion"]
            gest = r["gesture"]
            
            box_col = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_col, 2)
            lbl = f"{name} | {emo} | {gest}"
            cv2.putText(frame, lbl, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_col, 2)

        cv2.imshow("SOCIAL SCENE TEST", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
