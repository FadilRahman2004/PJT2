"""
Module 6 — Gesture & Pose Detection (Phase 3.95 — Holistic Rewrite)

Architecture:
  • YOLOv8-pose for multi-person tracking + bounding boxes
  • Per-person MediaPipe Holistic (face + pose + both hands, single model)
  • Body-posture check runs FIRST; hand classifier only fires if body is "free"
  • Normalised-coordinate thresholds (resolution-independent)
  • Sustained-Hold Gate: gesture must persist N frames before being reported
"""

import os
import time

import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

import config


# ─── One Euro Filter ────────────────────────────────────────────────────────────
class OneEuroFilter:
    """Adaptive low-pass filter for noisy 2-D signals."""

    def __init__(self, t0, x0, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta       = float(beta)
        self.d_cutoff   = float(d_cutoff)
        self.x_prev     = np.asarray(x0, dtype=float)
        self.dx_prev    = np.zeros_like(self.x_prev)
        self.t_prev     = t0

    def _alpha(self, cutoff, dt):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / max(dt, 1e-6))

    def __call__(self, t, x):
        dt = t - self.t_prev
        if dt <= 0:
            return self.x_prev
        dx = (x - self.x_prev) / dt
        edx   = self._alpha(self.d_cutoff, dt)
        dx_h  = edx * dx + (1 - edx) * self.dx_prev
        cut   = self.min_cutoff + self.beta * np.linalg.norm(dx_h)
        ex    = self._alpha(cut, dt)
        x_h   = ex * x + (1 - ex) * self.x_prev
        self.x_prev  = x_h
        self.dx_prev = dx_h
        self.t_prev  = t
        return x_h


# ─── Sustained-Hold Gate ────────────────────────────────────────────────────────
class SustainedGate:
    """Report a gesture only after it persists for `min_hold` consecutive frames."""

    def __init__(self, min_hold: int = 8):
        self.min_hold  = min_hold
        self._cand     = "Standing"
        self._streak   = 0
        self._out      = "Standing"

    def update(self, candidate: str) -> str:
        if candidate == self._cand:
            self._streak += 1
        else:
            self._cand   = candidate
            self._streak = 1
        if self._streak >= self.min_hold:
            self._out = self._cand
        return self._out


# ─── Hand Shape Classifier (ML/HaGRID) ──────────────────────────────────────────
# We now use the standard MediaPipe GestureRecognizer task which operates on
# image crops rather than manual angle calculation.
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
# ─── Main GestureDetector ───────────────────────────────────────────────────────
class GestureDetector:
    """
    Holistic-based gesture detector.

    Pipeline per frame per person:
      1. YOLOv8 → skeleton pixel coords + bounding box
      2. Crop person region → MediaPipe Holistic (face + pose + hands)
      3. Check body postures FIRST (arms crossed, hips, handshake, raised hand)
      4. ONLY if body is free → classify hand shape
      5. Head gestures from face-mesh nose landmarks (normalised)
      6. Sustained-Hold Gate → confirmed gesture
    """

    SUSTAIN_FRAMES    = 8      # consecutive frames to confirm a gesture
    # Normalised thresholds (0-1 space, resolution-independent)
    NOD_NORM_THR      = config.NOD_NORM_THRESHOLD    # 0.018
    SHAKE_NORM_THR    = config.SHAKE_NORM_THRESHOLD  # 0.022
    HEAD_AXIS_RATIO   = 3.0    # dominant axis must be N× the cross axis
    HEAD_MIN_REV      = 2      # minimum oscillation reversals

    def __init__(self, model_path=None, conf_threshold=None, history_len=None):
        model_path     = model_path     or config.POSE_MODEL
        conf_threshold = conf_threshold or config.GESTURE_CONFIDENCE_THRESHOLD
        history_len    = history_len    or config.GESTURE_HISTORY_FRAMES

        print(f"[GestureDetector] Loading YOLOv8-pose: {model_path}...")
        self.yolo      = YOLO(model_path)
        self.conf_thr  = conf_threshold
        self.hist_len  = history_len

        # ── MediaPipe Holistic ───────────────────────────────────────────────
        print("[GestureDetector] Initialising MediaPipe Holistic...")
        self._mp_holistic = mp.solutions.holistic
        self.holistic = self._mp_holistic.Holistic(
            model_complexity            = config.HOLISTIC_COMPLEXITY,
            smooth_landmarks            = True,
            min_detection_confidence    = 0.5,
            min_tracking_confidence     = 0.5,
            enable_segmentation         = False,
        )

        print("[GestureDetector] Initialising MediaPipe ML Gesture Recognizer...")
        task_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "gesture_recognizer.task")
        )
        if not os.path.exists(task_path):
            task_path = "gesture_recognizer.task"

        try:
            with open(task_path, "rb") as fh:
                model_data = fh.read()
            base_opts = python.BaseOptions(model_asset_buffer=model_data)
        except Exception as exc:
            print(f"[GestureDetector] Gesture model buffer load error: {exc}. Falling back.")
            base_opts = python.BaseOptions(model_asset_path=task_path)

        options = vision.GestureRecognizerOptions(
            base_options=base_opts,
            num_hands=2,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6
        )
        self.recognizer = vision.GestureRecognizer.create_from_options(options)

        # Per-track state
        self.history : dict = {} # {track_id: [keypoints]}
        self.filters : dict = {} # {track_id: {k_idx: OneEuroFilter}}
        self.gates   : dict = {} # {track_id: SustainedGate}
        self.nose_norm_hist: dict = {} # {track_id: [norm_y]}
        self.g_frame_counter: dict = {} # {track_id: count}
        self.last_h_result: dict = {} # {track_id: last_holistic}

        print("[GestureDetector] Holistic ready (body-first priority).")

    # ────────────────────────────────────────────────────────────────────────────
    def detect(self, frame: np.ndarray, person_ids=None):
        h_frame, w_frame = frame.shape[:2]

        # 1. YOLO pose detection/tracking
        results = self.yolo.track(
            frame, persist=True, verbose=False, conf=self.conf_thr
        )

        detections = []
        for r in results:
            if r.keypoints is None:
                continue
            yolo_ids = (
                r.boxes.id.cpu().numpy().astype(int).tolist()
                if r.boxes.id is not None
                else list(range(len(r.keypoints.data)))
            )
            boxes = r.boxes.xyxy.cpu().numpy()

            for i, (tid, kpts_t) in enumerate(zip(yolo_ids, r.keypoints.data)):
                kpts = kpts_t.cpu().numpy()   # (17, 3) in pixel space

                # Init per-track state
                if tid not in self.history:
                    self.history[tid]       = []
                    self.filters[tid]       = {}
                    self.gates[tid]         = SustainedGate(self.SUSTAIN_FRAMES)
                    self.nose_norm_hist[tid] = []
                    self.g_frame_counter[tid] = 0
                    self.last_h_result[tid] = None

                # One Euro Filter (pixel keypoints)
                now   = time.time()
                fkpts = np.copy(kpts)
                for k in range(len(fkpts)):
                    if fkpts[k, 2] > self.conf_thr:
                        if k not in self.filters[tid]:
                            self.filters[tid][k] = OneEuroFilter(
                                now, fkpts[k, :2],
                                min_cutoff=config.ONE_EURO_MIN_CUTOFF,
                                beta=config.ONE_EURO_BETA,
                            )
                        fkpts[k, :2] = self.filters[tid][k](now, fkpts[k, :2])

                self.history[tid].append(fkpts)
                if len(self.history[tid]) > self.hist_len:
                    self.history[tid].pop(0)

                # 2. Crop person region and run Holistic (with throttling)
                bbox    = boxes[i] if i < len(boxes) else None
                
                # Throttling logic
                self.g_frame_counter[tid] += 1
                skip = config.GESTURE_SKIP_FRAMES
                
                # Always run if person is new OR if we have significant motion
                is_moving = False
                if len(self.history[tid]) > 2:
                    # Simple motion check: wrist displacement
                    prev = self.history[tid][-2][9:11, :2] # L/R wrist
                    curr = self.history[tid][-1][9:11, :2]
                    dist = np.linalg.norm(curr - prev)
                    if dist > 0.02: is_moving = True

                should_run = (self.g_frame_counter[tid] % skip == 0) or is_moving or (self.last_h_result[tid] is None)

                if should_run:
                    h_result = self._run_holistic(frame, bbox, h_frame, w_frame)
                    self.last_h_result[tid] = h_result
                else:
                    h_result = self.last_h_result[tid]

                # 3. Update normalised nose history from Holistic face
                self._update_nose_norm(tid, h_result)

                # 4. Compute raw gesture candidate
                raw       = self._raw_gesture(tid, fkpts, h_result, frame, w_frame, h_frame)
                confirmed = self.gates[tid].update(raw)

                detections.append({
                    "id":         tid,
                    "keypoints":  kpts,
                    "holistic":   h_result,   # Expose raw Holistic data for drawing
                    "bbox_crop":  bbox,       # Needed to un-crop holistic coords
                    "gesture":    confirmed,
                    "_candidate": raw,
                })

        return detections

    # ────────────────────────────────────────────────────────────────────────────
    def _run_holistic(self, frame, bbox, h_frame, w_frame):
        """Crop person bounding box and run MediaPipe Holistic."""
        if bbox is not None:
            scale = 0.15      # expand bbox slightly
            x1 = max(0, int(bbox[0] - (bbox[2]-bbox[0]) * scale))
            y1 = max(0, int(bbox[1] - (bbox[3]-bbox[1]) * scale))
            x2 = min(w_frame, int(bbox[2] + (bbox[2]-bbox[0]) * scale))
            y2 = min(h_frame, int(bbox[3] + (bbox[3]-bbox[1]) * scale))
            crop = frame[y1:y2, x1:x2]
        else:
            crop = frame

        if crop.size == 0:
            return None

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        return self.holistic.process(rgb)

    # ────────────────────────────────────────────────────────────────────────────
    def _is_active_zone(self, wr, nose, l_hip, r_hip, sh_w) -> bool:
        """
        Check if a wrist is in the 'Active Gesture Zone'.
        1. Must be above the hips (not resting in lap)
        2. Must be away from the face/head (not scratching head/combing hair)
        """
        hip_y = max(l_hip[1], r_hip[1]) # Lowest hip point (more forgiving)
        if wr[1] > hip_y + 30:          # Too low (resting below hips)
            return False

        dist_to_nose = np.linalg.norm(wr[:2] - nose[:2])
        if dist_to_nose < sh_w * 0.35:  # Too close to face (lowered from 0.45)
            return False

        return True

    def _run_ml_recognizer(self, frame, wr, w_frame, h_frame, sh_w):
        """Run ML Hand Gesture Recognizer on full frame -> match to Active Zone wrist."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = self.recognizer.recognize(mp_img)

        if not res.gestures or not res.hand_landmarks:
            return "Generic"

        for i, hand_lm in enumerate(res.hand_landmarks):
            # ML wrist is landmark 0
            ml_wr_x = int(hand_lm[0].x * w_frame)
            ml_wr_y = int(hand_lm[0].y * h_frame)
            
            # Check if this ML wrist is near our holistic wrist (same hand)
            dist = np.linalg.norm(np.array([ml_wr_x, ml_wr_y]) - wr[:2])
            if dist < sh_w * 0.6: # Flexible matching distance

                top_gesture = res.gestures[i][0]
                cat = top_gesture.category_name
                # Map HaGRID labels ONLY to the requested core gestures
                mapping = {
                    "Thumb_Up": "Thumbs Up",
                    "Thumb_Down": "Thumbs Down",
                    "Pointing_Up": "Pointing",
                }
                mapped = mapping.get(cat, "Generic")
                if mapped != "Generic" and top_gesture.score > 0.4:  # ML Confidence
                    if mapped == "Pointing":
                        # Check Pointing direction (using ML landmarks 0, 5, 8)
                        # wrist=0, index_mcp=5, index_tip=8
                        mcp_x = hand_lm[5].x
                        tip_x = hand_lm[8].x
                        if tip_x > mcp_x + 0.05: return "Point Right"
                        if tip_x < mcp_x - 0.05: return "Point Left"
                        return "Point Forward"
                    return mapped
        return "Generic"

    # ────────────────────────────────────────────────────────────────────────────
    def _update_nose_norm(self, tid: int, h_result):
        """Store normalised nose position from Holistic face mesh."""
        hist = self.nose_norm_hist[tid]
        if h_result and h_result.face_landmarks:
            nose_lm = h_result.face_landmarks.landmark[1]  # tip of nose
            hist.append((nose_lm.x, nose_lm.y))
        
        if len(hist) > self.hist_len:
            hist.pop(0)

    # ────────────────────────────────────────────────────────────────────────────
    def _raw_gesture(self, tid: int, kpts: np.ndarray, h_result,
                     full_frame: np.ndarray, w_frame: int, h_frame: int) -> str:
        hist = self.history[tid]
        if not hist:
            return "Standing"

        def cp(kp): return kp[2] > self.conf_thr  # conf predicate

        l_sh  = kpts[5];  r_sh  = kpts[6]
        l_el  = kpts[7];  r_el  = kpts[8]
        l_wr  = kpts[9];  r_wr  = kpts[10]
        l_hip = kpts[11]; r_hip = kpts[12]
        nose  = kpts[0]

        sh_w = max(np.linalg.norm(l_sh[:2] - r_sh[:2]), 60.0)

        def reversals(arr):
            d = np.diff(arr)
            s = np.sign(d[d != 0])
            return int(np.sum(s[:-1] != s[1:])) if len(s) > 0 else 0

        # ════════════════════════════════════════════════════════════════════════
        #  TIER 1 — Body posture checks
        # ════════════════════════════════════════════════════════════════════════

        # ── Handshake Offer ───────────────────────────────────────────────────
        # Wrist extended forward (between shoulders horizontally) at chest/stomach height
        for wr, sh, h_hand in [(l_wr, l_sh, h_result.left_hand_landmarks if h_result else None), 
                               (r_wr, r_sh, h_result.right_hand_landmarks if h_result else None)]:
            if cp(wr) and cp(l_sh) and cp(r_sh):
                # Is wrist horizontally between the left and right shoulder?
                sh_left_x = max(l_sh[0], r_sh[0])
                sh_right_x = min(l_sh[0], r_sh[0])
                
                if sh_right_x - 30 < wr[0] < sh_left_x + 30: 
                    # Height restrictions: below nose, above waist
                    waist_y = max(l_sh[1], r_sh[1]) + (sh_w * 1.5)
                    if wr[1] > nose[1] + 20 and wr[1] < waist_y:
                        # Must be somewhat extended from the shoulder
                        dist = np.linalg.norm(wr[:2] - sh[:2])
                        if dist > sh_w * 0.35:
                            # 🛡️ HARDENING: Vertical palm check
                            if h_hand:
                                lm = h_hand.landmark
                                # Thumb base (5) vs Pinky base (17)
                                dx = abs(lm[5].x - lm[17].x)
                                dy = abs(lm[5].y - lm[17].y)
                                if dy > dx: # Palm is sideways (handshake orientation)
                                    return "Handshake Offer"
                            # If no hand mesh detected, we don't confirm handshake
                            # to prevent "everything detects handshake" bug.


        # (Waving logic has been moved cleanly to the ML Hand Recognizer: Open_Palm)
        
        # ════════════════════════════════════════════════════════════════════════
        #  TIER 2 — Hand shape (ML Recognizer + Active Zone check)
        # ════════════════════════════════════════════════════════════════════════
        for wr in [l_wr, r_wr]:
            if cp(wr) and cp(nose) and cp(l_hip) and cp(r_hip):
                # 1. Zone Check
                if self._is_active_zone(wr, nose, l_hip, r_hip, sh_w):
                    # 2. ML Check (Full frame context)
                    shape = self._run_ml_recognizer(full_frame, wr, w_frame, h_frame, sh_w)
                    if shape != "Generic":
                        return shape

        # ════════════════════════════════════════════════════════════════════════
        #  TIER 3 — Head gestures (Pixel history)
        # ════════════════════════════════════════════════════════════════════════
        if len(hist) >= 15:
            nose_hist = [h[0] for h in hist if h[0][2] > self.conf_thr]
            if len(nose_hist) >= 12:
                xs = np.array([p[0] for p in nose_hist])
                ys = np.array([p[1] for p in nose_hist])
                travel_x = float(np.ptp(xs))
                travel_y = float(np.ptp(ys))
                std_x = float(np.std(xs))
                std_y = float(np.std(ys))

                def _head_reversals(arr):
                    d = np.diff(arr)
                    s = np.sign(d[d != 0])
                    return int(np.sum(s[:-1] != s[1:])) if len(s) > 0 else 0

                if (travel_y > 20 and std_y > std_x * self.HEAD_AXIS_RATIO and
                        _head_reversals(ys) >= self.HEAD_MIN_REV):
                    return "Nodding (Yes)"

                if (travel_x > 25 and std_x > std_y * self.HEAD_AXIS_RATIO and
                        _head_reversals(xs) >= self.HEAD_MIN_REV):
                    return "Shaking Head (No)"

        return "Standing"
