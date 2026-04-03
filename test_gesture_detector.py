"""
Test Script: Holistic Gesture Detection (Phase 3.95)
Shows candidate vs. confirmed gesture, pose skeleton, and raw Holistic data.
"""
import cv2
import numpy as np
import config
from modules.gesture_detector import GestureDetector
from modules.person_detector import PersonDetector

import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

GESTURE_COLORS = {
    "Standing":           (160, 160, 160),
    "Waving":             (0,   255, 180),
    "Nodding (Yes)":      (50,  200, 255),
    "Shaking Head (No)":  (0,   60,  255),
    "Thumbs Up":          (0,   255, 80),
    "Thumbs Down":        (0,   80,  255),
    "Point Left":         (255, 255, 0),
    "Point Right":        (255, 255, 0),
    "Point Forward":      (255, 255, 0),
    "Handshake Offer":    (100, 200, 255),
}

# YOLO pose edges to draw
SKELETON_EDGES = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


def draw_skeleton(frame, kpts, color):
    h, w = frame.shape[:2]
    for a, b in SKELETON_EDGES:
        if kpts[a, 2] > 0.4 and kpts[b, 2] > 0.4:
            pa = (int(kpts[a, 0]), int(kpts[a, 1]))
            pb = (int(kpts[b, 0]), int(kpts[b, 1]))
            cv2.line(frame, pa, pb, color, 2)
    for k in range(len(kpts)):
        if kpts[k, 2] > 0.4:
            cv2.circle(frame, (int(kpts[k, 0]), int(kpts[k, 1])), 4, color, -1)


def draw_holistic_landmarks(frame, bbox, h_result):
    if not h_result or bbox is None:
        return
    
    h_frame, w_frame = frame.shape[:2]
    scale = 0.15
    x1 = max(0, int(bbox[0] - (bbox[2]-bbox[0]) * scale))
    y1 = max(0, int(bbox[1] - (bbox[3]-bbox[1]) * scale))
    crop_w = min(w_frame, int(bbox[2] + (bbox[2]-bbox[0]) * scale)) - x1
    crop_h = min(h_frame, int(bbox[3] + (bbox[3]-bbox[1]) * scale)) - y1

    def un_crop(landmark_list):
        if not landmark_list: return None
        # Create a deep copy to shift coordinates back to full frame
        from google.protobuf.json_format import ParseDict, MessageToDict
        import copy
        lm_copy = copy.deepcopy(landmark_list)
        for lm in lm_copy.landmark:
            lm.x = (lm.x * crop_w + x1) / w_frame
            lm.y = (lm.y * crop_h + y1) / h_frame
        return lm_copy

    un_face = un_crop(h_result.face_landmarks)
    un_lh   = un_crop(h_result.left_hand_landmarks)
    un_rh   = un_crop(h_result.right_hand_landmarks)

    if un_face:
        mp_drawing.draw_landmarks(
            frame, un_face, mp_holistic.FACEMESH_TESSELATION,
            mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
        )
    if un_lh:
        mp_drawing.draw_landmarks(frame, un_lh, mp_holistic.HAND_CONNECTIONS)
    if un_rh:
        mp_drawing.draw_landmarks(frame, un_rh, mp_holistic.HAND_CONNECTIONS)


def main():
    print("[Test] Holistic Gesture Detector — Body-First Priority")
    detector = GestureDetector()
    cap = cv2.VideoCapture(0)
    print("Controls: [Q] Quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gestures = detector.detect(frame)

        for g in gestures:
            confirmed = g["gesture"]
            candidate = g.get("_candidate", "?")
            kpts      = g["keypoints"]
            h_result  = g.get("holistic", None)
            bbox      = g.get("bbox_crop", None)
            color     = GESTURE_COLORS.get(confirmed, (200, 200, 200))

            draw_skeleton(frame, kpts, color)
            draw_holistic_landmarks(frame, bbox, h_result)

            nose = kpts[0]
            nx, ny = int(nose[0]), int(nose[1])

            cv2.putText(frame, confirmed,
                        (nx - 70, ny - 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(frame, f"[Cand: {candidate}]",
                        (nx - 70, ny - 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (130, 130, 130), 1)
            cv2.putText(frame, "[ML Recognizer + Zone Gating]",
                        (nx - 70, ny - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1)

        cv2.imshow("ML Gesture Recognizer Test (Phase 4.0)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
