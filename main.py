"""
main.py — Multi-Modal Real-Time AI Assistant  (Phase 1+2)
==========================================================
Orchestrates the full pipeline:

    Webcam → Person Detection → Face Detection & Alignment
           → Face Recognition + Emotion Detection + Gesture Detection → TTS Narration

Threads:
    1. Frame Capture        → frame_queue
    2. Detection + Recog.   → result_queue
    3. TTS Narration        (reads result_queue)
    Main thread             → OpenCV display loop / user input

Controls:
    [R]  Register an unknown person
    [Q]  Quit the application
"""

import sys
import os
import time
import threading
import queue

import cv2
import numpy as np

# ── project imports ───────────────────────────
import config
from modules.frame_capture import FrameCapture
from modules.person_detector import PersonDetector
from modules.face_detector import FaceDetector
from modules.face_recognizer import FaceRecognizer
from modules.emotion_detector import EmotionDetector
from modules.gesture_detector import GestureDetector
from modules.context_engine import ContextEngine
from modules.narrative_engine import NarrativeEngine
from modules.tts_engine import TTSEngine
from modules.stt_engine import STTEngine
from modules.registration import RegistrationManager
from modules.llm_engine import LLMEngine
from utils.drawing import draw_person_box, draw_info_overlay, draw_pose_keypoints

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def ensure_data_dir():
    """Create the data directory if it doesn't exist."""
    os.makedirs(config.DATA_DIR, exist_ok=True)


# ──────────────────────────────────────────────
# Detection + Recognition Worker
# ──────────────────────────────────────────────

def detection_worker(frame_queue, result_queue, person_detector,
                     face_detector, face_recognizer, emotion_detector,
                     gesture_detector, skip_frames):
    """
    Thread: reads frames → detects persons → recognises faces + emotions + gestures.
    Pushes annotated results into result_queue.
    """
    frame_idx = 0
    face_align_counters = {} # {tid: count}
    
    while True:
        item = frame_queue.get()
        if item is None:
            result_queue.put(None)
            break

        frame = item
        frame_idx += 1
        if frame_idx % skip_frames != 0:
            # On skipped frames, push frame with empty detections
            result_queue.put({"frame": frame, "persons": []})
            continue

        # 1. Detection AND Tracking (Native ByteTrack/BoT-SORT)
        # Returns: [{"person_id": T_ID, "bbox": (x1,y1,x2,y2), ...}]
        persons_detected = person_detector.detect(frame)
        
        # 2. Get Pose/Gestures matching those IDs
        person_track_ids = [p["person_id"] for p in persons_detected]
        gestures = gesture_detector.detect(frame, person_ids=person_track_ids)
        
        results = []

        for i, person in enumerate(persons_detected):
            bbox = person["bbox"]
            track_id = person["person_id"]

            # Match gesture by track_id
            best_gesture = "Standing"
            best_kpts = None
            for g in gestures:
                if g["id"] == track_id:
                    best_gesture = g["gesture"]
                    best_kpts = g["keypoints"]
                    break

            # Distance Gating: Skip Face/Emotion if too far
            x1, y1, x2, y2 = bbox
            if (x2 - x1) < config.MIN_PROXIMITY_WIDTH:
                aligned_face = None
            else:
                # 3. Face Alignment Throttling: Only align every 5 frames
                face_align_counters[track_id] = face_align_counters.get(track_id, 0) + 1
                if face_align_counters[track_id] % 5 == 0 or track_id not in face_recognizer.identity_cache:
                    body_crop = person["cropped_body"]
                    aligned_face, face_bbox = face_detector.detect_and_align(body_crop)
                else:
                    aligned_face = None

            # 4. Recognize & Detect (Handles aligned_face=None via Cache)
            identity = face_recognizer.recognize(aligned_face, person_id=track_id)
            emotion = emotion_detector.detect(aligned_face, person_id=track_id)

            # Auto-Enrollment Logic
            if identity["name"] == "Unknown" and identity.get("embedding") is not None:
                new_name = face_recognizer.auto_enroll(track_id, identity["embedding"])
                if new_name:
                    identity["name"] = new_name
                    # Signal narrative engine about new enrollment
                    print(f"[Self-Learning] Auto-enrolled: {new_name}")

            results.append({
                "person_id": track_id, # Link results to track_id
                "bbox": bbox,
                "face_aligned": aligned_face,
                "identity": identity,
                "emotion": emotion,
                "gesture": best_gesture,
                "keypoints": best_kpts,
            })

        result_queue.put({"frame": frame, "persons": results})


# ──────────────────────────────────────────────
# Narration Worker
# ──────────────────────────────────────────────

def narration_worker(narration_queue, tts):
    """
    Thread: reads narration strings and speaks them.
    """
    while True:
        text = narration_queue.get()
        if text is None:
            break
        tts.speak(text)


# ──────────────────────────────────────────────
# Registration Flow (runs in main thread)
# ──────────────────────────────────────────────

def run_registration(reg_manager, frame_queue_ref, face_detector,
                     face_recognizer, capture):
    """
    Interactive registration: capture N frames, get name, store.
    Returns True if registration succeeded.
    """
    print("\n" + "=" * 50)
    print("  FACE REGISTRATION")
    print("=" * 50)
    name = input("Enter the person's name (or 'cancel'): ").strip()
    if not name or name.lower() == "cancel":
        print("[Registration] Cancelled.")
        reg_manager.reset()
        return False

    reg_manager.start_registration()
    print(f"[Registration] Capturing {reg_manager.capture_count} "
          f"face frames for '{name}'...")
    print("Look at the camera…")

    embeddings = []
    attempts = 0
    max_attempts = reg_manager.capture_count * 5  # allow some failures

    while len(embeddings) < reg_manager.capture_count and attempts < max_attempts:
        frame = capture.read(timeout=1.0)
        if frame is None:
            attempts += 1
            continue

        # Simple face detection on full frame for registration
        aligned, _ = face_detector.detect_and_align(frame)
        if aligned is not None:
            emb = reg_manager.collect_embedding(aligned)
            if emb is not None:
                embeddings.append(emb)
                print(f"  Captured {len(embeddings)}/"
                      f"{reg_manager.capture_count}")
                # Add delay to ensure diverse embeddings (different angles/lighting)
                time.sleep(0.3)
        attempts += 1

    if embeddings:
        reg_manager.finish_registration(name, embeddings)
        print(f"[Registration] ✓ '{name}' registered successfully!")
        return True
    else:
        print("[Registration] ✗ No faces captured. Try again.")
        reg_manager.reset()
        return False


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    ensure_data_dir()

    print("Initializing modules…")

    # ── init modules ──────────────────────────
    capture = FrameCapture(
        camera_index=config.CAMERA_INDEX,
        width=config.FRAME_WIDTH,
        height=config.FRAME_HEIGHT,
        target_fps=config.TARGET_FPS,
        queue_size=config.FRAME_QUEUE_SIZE,
    )

    person_detector = PersonDetector(
        model_path=config.YOLO_MODEL,
        confidence=config.YOLO_CONFIDENCE,
        person_class=config.YOLO_PERSON_CLASS,
    )

    face_detector = FaceDetector(
        align_size=config.FACE_ALIGN_SIZE,
        scale_factor=config.HAAR_SCALE_FACTOR,
        min_neighbors=config.HAAR_MIN_NEIGHBORS,
        min_face_size=config.HAAR_MIN_FACE_SIZE,
    )

    face_recognizer = FaceRecognizer(
        pretrained=config.FACENET_PRETRAINED,
        db_path=config.FACE_DB_PATH,
        threshold=config.FACE_MATCH_THRESHOLD,
        max_embeddings=config.MAX_EMBEDDINGS_PER_PERSON,
        enroll_threshold=config.AUTO_ENROLL_THRESHOLD,
    )

    tts = TTSEngine(
        rate=config.TTS_RATE,
        volume=config.TTS_VOLUME,
        cooldown=config.TTS_COOLDOWN,
    )
    
    stt = STTEngine(
        use_stt=config.USE_STT,
        wake_word=config.STT_WAKE_WORD,
        timeout=config.STT_TIMEOUT,
        phrase_limit=config.STT_PHRASE_LIMIT,
        tts_engine=tts
    )

    emotion_detector = EmotionDetector()

    gesture_detector = GestureDetector(
        model_path=config.POSE_MODEL,
        conf_threshold=config.GESTURE_CONFIDENCE_THRESHOLD,
        history_len=config.GESTURE_HISTORY_FRAMES,
    )

    context_engine = ContextEngine(
        vibe_window_fps=config.TARGET_FPS // config.SKIP_FRAMES,
        window_seconds=config.VIBE_WINDOW_SECONDS
    )

    llm_engine = LLMEngine(
        api_key=config.LLM_API_KEY,
        model_name=config.LLM_MODEL_NAME,
        system_prompt=config.LLM_SYSTEM_PROMPT
    )

    narrative_engine = NarrativeEngine(
        style=config.NARRATION_STYLE,
        summary_cooldown=config.NARRATION_SUMMARY_COOLDOWN,
        llm_engine=llm_engine if config.USE_LLM else None
    )

    reg_manager = RegistrationManager(
        face_recognizer=face_recognizer,
        persist_frames=config.UNKNOWN_PERSIST_FRAMES,
        capture_count=config.REGISTRATION_CAPTURE_COUNT,
    )

    print("All modules loaded. Starting pipeline…")

    # ── queues ────────────────────────────────
    frame_queue = queue.Queue(maxsize=config.FRAME_QUEUE_SIZE)
    result_queue = queue.Queue(maxsize=config.FRAME_QUEUE_SIZE)
    narration_queue = queue.Queue()

    # ── start threads ─────────────────────────
    capture.start()
    tts.start()
    stt.start()

    det_thread = threading.Thread(
        target=detection_worker,
        args=(frame_queue, result_queue, person_detector,
              face_detector, face_recognizer, emotion_detector,
              gesture_detector, config.SKIP_FRAMES),
        daemon=True,
    )
    det_thread.start()

    nar_thread = threading.Thread(
        target=narration_worker,
        args=(narration_queue, tts),
        daemon=True,
    )
    nar_thread.start()

    # ── display / main loop ───────────────────
    last_narration = {}  # name → timestamp  (extra cooldown per-person)
    display_frame = None

    print(f"\nCamera open. Press [R] to register, [Q] to quit.\n")

    try:
        while True:
            # Push frames into detection pipeline
            raw_frame = capture.read(timeout=0.5)
            if raw_frame is not None:
                if frame_queue.full():
                    try:
                        frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                frame_queue.put(raw_frame)

            # Pull detection results
            try:
                result = result_queue.get_nowait()
            except queue.Empty:
                result = None

            # Process User Voice Queries
            user_query = stt.get_query()
            if user_query and result:
                # We have a question and a current context frame
                print(f"[Main] Processing query: '{user_query}'")
                persons = result["persons"]
                context_engine.update(persons)
                summary = context_engine.get_summary()
                interactions = context_engine.detect_interactions(persons)
                
                llm_context = {
                    "global_vibe": summary["vibe"],
                    "people_present": [
                        {
                            "name": p["identity"]["name"],
                            "emotion": p.get("emotion", {}).get("emotion", "Neutral"),
                            "gesture": p.get("gesture", "Standing")
                        } for p in persons
                    ],
                    "social_interactions": interactions,
                    "timestamp": time.strftime("%H:%M:%S")
                }
                
                answer = llm_engine.answer_user_query(llm_context, user_query)
                if answer:
                    print(f"  -> Assistant Says: {answer}")
                    narration_queue.put(answer)
                
                # Reset Narrative cooldown so it doesn't try to proactively 
                # narrate right after answering a question.
                narrative_engine.last_summary_time = time.time()
                
            elif result is not None:
                frame = result["frame"]
                persons = result["persons"]
                has_unknown = False

                # 1. Update Context and Narrative
                context_engine.update(persons)
                summary = context_engine.get_summary()
                interactions = context_engine.detect_interactions(persons)
                
                # Generate proactive narration
                proactive_text = narrative_engine.generate_narration(
                    persons, summary, interactions
                )
                if proactive_text:
                    narration_queue.put(proactive_text)

                for p in persons:
                    name = p["identity"]["name"]
                    conf = p["identity"]["confidence"]
                    bbox = p["bbox"]
                    emotion = p.get("emotion", {})
                    emo_label = emotion.get("emotion", "")
                    emo_conf = emotion.get("confidence", 0.0)
                    gesture = p.get("gesture", "Standing")
                    kpts = p.get("keypoints")

                    if name != "Unknown":
                        color = config.BOX_COLOR_KNOWN
                        # Build narration with emotion and gesture
                        now = time.time()
                        last = last_narration.get(name, 0)
                        if now - last > config.TTS_COOLDOWN:
                            msg = f"{name}"
                            if (emo_label and emo_label != "Neutral"
                                    and emo_conf >= config.EMOTION_CONFIDENCE_THRESHOLD):
                                msg += f" looks {emo_label.lower()}"
                            
                            if gesture and gesture != "Standing":
                                # Conjunction for natural speech
                                clean_gest = gesture.lower()
                                if "point" in clean_gest:
                                    clean_gest = clean_gest.replace("point", "pointing")
                                elif "handshake" in clean_gest:
                                    clean_gest = "offering a handshake"
                                
                                if "looks" in msg:
                                    msg += f" and is {clean_gest}."
                                else:
                                    msg += f" is {clean_gest}."
                            else:
                                if "looks" in msg:
                                    msg += "."
                                else:
                                    msg += " is in front of you."
                                    
                            narration_queue.put(msg)
                            last_narration[name] = now
                    else:
                        color = config.BOX_COLOR_UNKNOWN
                        has_unknown = True

                    draw_person_box(frame, bbox, name, conf, color,
                                    config.FONT_SCALE, config.FONT_THICKNESS,
                                    emotion=emo_label,
                                    emotion_conf=emo_conf,
                                    gesture=gesture)
                    
                    if kpts is not None:
                        draw_pose_keypoints(frame, kpts)

                # Track unknown persistence
                if has_unknown:
                    if reg_manager.tick_unknown():
                        # Let registration be a high-priority narration
                        narration_queue.put(
                            "I see someone I don't recognize."
                        )
                else:
                    reg_manager.reset()

                draw_info_overlay(frame, capture.fps, len(persons), 
                                  vibe=summary["vibe"])
                display_frame = frame

            # Show window
            if config.SHOW_WINDOW and display_frame is not None:
                cv2.imshow(config.WINDOW_NAME, display_frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == ord("Q"):
                break
            elif key == ord("r") or key == ord("R"):
                run_registration(reg_manager, frame_queue,
                                 face_detector, face_recognizer, capture)

    except KeyboardInterrupt:
        print("\nShutting down…")

    # ── cleanup ───────────────────────────────
    frame_queue.put(None)      # signal detection thread to stop
    narration_queue.put(None)  # signal narration thread to stop
    det_thread.join(timeout=3)
    nar_thread.join(timeout=3)
    capture.stop()
    stt.stop()
    tts.stop()
    cv2.destroyAllWindows()
    print("Goodbye.")


if __name__ == "__main__":
    main()
