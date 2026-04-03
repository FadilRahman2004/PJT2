"""
Module 2 — Person Detection
Detects all people in a frame using YOLOv8.
"""

import numpy as np
from ultralytics import YOLO


class PersonDetector:
    """YOLOv8-based person detector."""

    def __init__(self, model_path="yolov8n.pt", confidence=0.5,
                 person_class=0):
        """
        Parameters
        ----------
        model_path : str
            Path or name of the YOLOv8 model weights.
        confidence : float
            Minimum detection confidence.
        person_class : int
            COCO class ID for 'person' (default 0).
        """
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.person_class = person_class

    def detect(self, frame):
        """
        Run detection AND tracking on *frame* and return a list of person dicts.

        Each dict contains:
            person_id  : int — persistent track ID from YOLO
            bbox       : (x1, y1, x2, y2) — pixel coordinates
            confidence : float
            cropped_body : np.ndarray — body crop from the frame
        """
        # USE PERSIST=TRUE for native ByteTrack/BoT-SORT
        results = self.model.track(frame, persist=True, verbose=False, conf=self.confidence)
        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is None or boxes.id is None:
                # Fallback to standard detect if tracking info is missing in some frames
                if boxes is not None:
                    for idx, box in enumerate(boxes):
                        if int(box.cls[0]) == self.person_class:
                            self._append_detection(detections, frame, box, idx)
                continue

            for box in boxes:
                cls_id = int(box.cls[0])
                if cls_id != self.person_class:
                    continue

                track_id = int(box.id[0])
                self._append_detection(detections, frame, box, track_id)

        return detections

    def _append_detection(self, detections, frame, box, person_id):
        """Helper to format detection dictionary."""
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 > x1 and y2 > y1:
            cropped_body = frame[y1:y2, x1:x2].copy()
            detections.append({
                "person_id": person_id,
                "bbox": (x1, y1, x2, y2),
                "confidence": conf,
                "cropped_body": cropped_body,
            })
