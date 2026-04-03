"""
Drawing Utilities
Bounding-box overlays, labels, and status bar for the OpenCV preview window.
"""

import cv2


def draw_person_box(frame, bbox, name="Unknown", confidence=0.0,
                    color=(0, 255, 0), font_scale=0.6, thickness=2,
                    emotion=None, emotion_conf=0.0, gesture=None):
    """
    Draw a labelled bounding box around a detected person.

    Parameters
    ----------
    frame : np.ndarray
        BGR image to draw on (modified in-place).
    bbox : tuple
        (x1, y1, x2, y2) pixel coordinates.
    name : str
        Identity label.
    confidence : float
        Recognition confidence (shown in label).
    color : tuple
        BGR colour.
    emotion : str | None
        Detected emotion label.
    emotion_conf : float
        Emotion detection confidence.
    """
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # Build identity label
    label = f"{name} ({confidence:.0%})" if name != "Unknown" else "Unknown"
    
    # Append emotion if significant
    if emotion and emotion != "Neutral" and emotion_conf > 0.3:
        label += f" | {emotion} ({emotion_conf:.0%})"
        
    # Append gesture
    if gesture and gesture != "Standing":
        label += f" | {gesture}"

    (tw, th), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (0, 0, 0), thickness, cv2.LINE_AA)


def draw_info_overlay(frame, fps=0.0, person_count=0, vibe="Quiet"):
    """
    Draw a translucent status bar at the top of the frame.
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 36), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    info = f"FPS: {fps:.1f}  |  People: {person_count}  |  Vibe: {vibe}  |  [R] Reg  [Q] Quit"
    cv2.putText(frame, info, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (200, 200, 200), 1, cv2.LINE_AA)


def draw_pose_keypoints(frame, kpts, conf_threshold=0.5):
    """
    Draw 17 COCO keypoints and connections.
    """
    # Connections: (idx1, idx2)
    connections = [
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # upper body
        (5, 11), (6, 12), (11, 12),              # torso
        (11, 13), (13, 15), (12, 14), (14, 16)   # lower body
    ]
    
    # Draw connections
    for start_idx, end_idx in connections:
        pt1 = kpts[start_idx]
        pt2 = kpts[end_idx]
        if pt1[2] > conf_threshold and pt2[2] > conf_threshold:
            cv2.line(frame, (int(pt1[0]), int(pt1[1])), 
                     (int(pt2[0]), int(pt2[1])), (255, 255, 0), 1)
            
    # Draw keypoints
    for i, pt in enumerate(kpts):
        if pt[2] > conf_threshold:
            cv2.circle(frame, (int(pt[0]), int(pt[1])), 3, (0, 255, 255), -1)
