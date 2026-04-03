"""
Module 3 — Face Detection & Alignment (OpenCV Haar Cascades)
Detects faces using OpenCV's built-in Haar cascade classifiers,
aligns them using eye landmarks, and resizes to 160×160 for FaceNet.

No extra dependencies — everything ships with OpenCV.
"""

import cv2
import numpy as np
import mediapipe as mp


class FaceDetector:
    """
    OpenCV Haar cascade face detector with eye-landmark alignment.
    Uses cv2.data.haarcascades which ship with opencv-python.
    """

    def __init__(self, align_size=(160, 160), scale_factor=1.1,
                 min_neighbors=5, min_face_size=(40, 40)):
        """
        Parameters
        ----------
        align_size : tuple
            (width, height) of aligned face output.
        scale_factor : float
            Haar cascade scale factor (1.05–1.3).
        min_neighbors : int
            Haar cascade min neighbours (higher = fewer false positives).
        min_face_size : tuple
            Minimum face size to detect.
        """
        self.align_size = align_size
        
        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1, # 1 for far range (within 5m), 0 for close range (within 2m)
            min_detection_confidence=0.5
        )

    def detect_and_align(self, image):
        """
        Detect face using MediaPipe, align with landmarks, and return (aligned, bbox).
        """
        if image is None or image.size == 0:
            return None, None

        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(img_rgb)

        if not results.detections:
            return None, None

        # Pick largest face
        ih, iw = image.shape[:2]
        best_detect = max(results.detections, key=lambda d: d.location_data.relative_bounding_box.width)
        
        bbox = best_detect.location_data.relative_bounding_box
        x1 = int(bbox.xmin * iw)
        y1 = int(bbox.ymin * ih)
        x2 = int((bbox.xmin + bbox.width) * iw)
        y2 = int((bbox.ymin + bbox.height) * ih)
        
        # Clamp
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(iw, x2), min(ih, y2)
        
        # Landmarks for alignment
        landmarks = best_detect.location_data.relative_keypoints
        # MediaPipe Keypoints: 0:Right Eye, 1:Left Eye, 2:Nose, 3:Mouth, 4:Right Ear, 5:Left Ear
        try:
            r_eye = np.array([landmarks[0].x * iw, landmarks[0].y * ih])
            l_eye = np.array([landmarks[1].x * iw, landmarks[1].y * ih])
            nose = np.array([landmarks[2].x * iw, landmarks[2].y * ih])
            
            # Use landmarks to align
            aligned = self._align_with_landmarks(image, l_eye, r_eye, nose, x1, y1, x2-x1, y2-y1)
            return aligned, (x1, y1, x2, y2)
        except Exception:
            # Fallback
            return self._simple_crop(image, x1, y1, x2-x1, y2-y1), (x1, y1, x2, y2)

    def _align_with_landmarks(self, image, l_eye, r_eye, nose, fx, fy, fw, fh):
        """Rotation/scaling alignment using facial keypoints."""
        dy = r_eye[1] - l_eye[1]
        dx = r_eye[0] - l_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))

        # Rotate around nose tip
        M = cv2.getRotationMatrix2D((float(nose[0]), float(nose[1])), float(angle), 1.0)
        ih, iw = image.shape[:2]
        rotated = cv2.warpAffine(image, M, (ih, ih), flags=cv2.INTER_CUBIC)

        # Better padding for deep models
        pad_w = int(fw * 0.2)
        pad_h = int(fh * 0.2)
        cx1 = max(0, fx - pad_w)
        cy1 = max(0, fy - pad_h)
        cx2 = min(iw, fx + fw + pad_w)
        cy2 = min(ih, fy + fh + pad_h)

        crop = rotated[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return None

        # Ensure output is RGB for FaceNet
        face_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        return cv2.resize(face_rgb, self.align_size)

    def _simple_crop(self, image, fx, fy, fw, fh):
        """Fallback: crop face with padding and resize, no rotation."""
        ih, iw = image.shape[:2]
        pad_w = int(fw * 0.2)
        pad_h = int(fh * 0.2)
        cx1 = max(0, fx - pad_w)
        cy1 = max(0, fy - pad_h)
        cx2 = min(iw, fx + fw + pad_w)
        cy2 = min(ih, fy + fh + pad_h)

        crop = image[cy1:cy2, cx1:cx2]
        face_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        return cv2.resize(face_rgb, self.align_size)
