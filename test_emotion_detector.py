"""
Test Script: Hybrid Emotion Detection (HuggingFace ViT + Geometric Mesh)
Shows emotion label, confidence, all-class score bars, and
highlights the geometric correction multipliers.
"""
import cv2
import numpy as np
import config
from modules.person_detector import PersonDetector
from modules.face_detector import FaceDetector
from modules.emotion_detector import EmotionDetector

COLORS = {
    "Happy":   (0,   220, 0),
    "Surprise":(0,   180, 255),
    "Neutral": (200, 200, 200),
    "Sad":     (255, 100, 50),
    "Angry":   (0,   0,   255),
    "Fear":    (0,   100, 200),
    "Disgust": (50,  200, 50),
    "Contempt":(180, 50,  220),
}

def draw_score_bars(frame, scores, top_left):
    x, y = top_left
    for i, (lbl, s) in enumerate(sorted(scores.items(), key=lambda x: -x[1])):
        col = COLORS.get(lbl, (200, 200, 200))
        bar_len = int(s * 120)
        cv2.rectangle(frame, (x, y + i * 18), (x + bar_len, y + i * 18 + 14), col, -1)
        cv2.putText(frame, f"{lbl[:3]}: {s:.2f}", (x + bar_len + 4, y + i * 18 + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1)

def main():
    print("[Test] Initializing Hybrid Emotion Suite (HuggingFace + Mesh)...")
    person_det = PersonDetector(confidence=0.5)
    face_det   = FaceDetector()
    emotion_det = EmotionDetector()

    cap = cv2.VideoCapture(0)
    print("Controls: [Q] Quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        persons = person_det.detect(frame)

        for p in persons:
            body_crop = p["cropped_body"]
            tid       = p["person_id"]
            aligned, face_bbox = face_det.detect_and_align(body_crop)

            if aligned is not None:
                emo = emotion_det.detect(aligned, person_id=tid)
                label    = emo["emotion"]
                conf     = emo["confidence"]
                scores   = emo.get("scores", {})
                box_col  = COLORS.get(label, (200, 200, 200))

                # Face bounding box on main frame
                bx1, by1, _, _ = p["bbox"]
                fx1, fy1, fx2, fy2 = face_bbox
                gx1, gy1 = bx1 + fx1, by1 + fy1
                gx2, gy2 = bx1 + fx2, by1 + fy2

                cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), box_col, 2)
                cv2.putText(frame, f"ID {tid}: {label} ({conf:.2f})",
                            (gx1, gy1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, box_col, 2)

                # Score bars bottom-left of face
                if scores:
                    draw_score_bars(frame, scores, (gx1, gy2 + 4))

                # Small face preview window
                face_viz = cv2.resize(cv2.cvtColor(aligned, cv2.COLOR_RGB2BGR), (160, 160))
                cv2.imshow(f"Face_ID_{tid}", face_viz)

        cv2.imshow("Hybrid Emotion Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
