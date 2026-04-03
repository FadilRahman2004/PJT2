"""
analyze_video.py — Video File Analysis Mode
============================================
Runs the full AI pipeline (Person Detection, Face Recognition,
Emotion Detection, Gesture Analysis, Vibe Engine) on a
pre-recorded video file instead of a live webcam stream.

Usage:
    python analyze_video.py                         # opens a file dialog
    python analyze_video.py --video path/to/vid.mp4 # direct path
    python analyze_video.py --video vid.mp4 --no-tts --speed 2.0

Controls:
    [Q]   Quit
    [Space] Pause / Resume
    [S]   Save current frame as screenshot
"""

import sys
import os
import argparse
import time

import cv2
import numpy as np

# ── project imports ───────────────────────────
import config
from modules.person_detector import PersonDetector
from modules.face_detector import FaceDetector
from modules.face_recognizer import FaceRecognizer
from modules.emotion_detector import EmotionDetector
from modules.gesture_detector import GestureDetector
from modules.context_engine import ContextEngine
from modules.narrative_engine import NarrativeEngine
from modules.tts_engine import TTSEngine
from modules.llm_engine import LLMEngine
from utils.drawing import draw_person_box, draw_info_overlay, draw_pose_keypoints


def pick_video_file():
    """
    Opens a file-picker dialog to choose a video file.
    Falls back to a command-line prompt if tkinter is unavailable.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(
            title="Select a video file",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.webm"),
                ("All files", "*.*")
            ]
        )
        root.destroy()
        return path if path else None
    except Exception:
        path = input("Enter path to video file: ").strip().strip('"')
        return path if os.path.isfile(path) else None


def draw_progress_bar(frame, current_frame, total_frames):
    """Draw a thin progress bar at the bottom of the frame."""
    h, w = frame.shape[:2]
    bar_h = 5
    if total_frames > 0:
        filled = int(w * current_frame / total_frames)
        cv2.rectangle(frame, (0, h - bar_h), (w, h), (50, 50, 50), -1)
        cv2.rectangle(frame, (0, h - bar_h), (filled, h), (0, 210, 100), -1)


def run_analysis(video_path, use_tts=True, playback_speed=1.0, save_output=True):
    """
    Core analysis loop for a video file.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\n[Video] {os.path.basename(video_path)}")
    print(f"  Resolution : {frame_w}x{frame_h}")
    print(f"  FPS        : {fps:.1f}")
    print(f"  Frames     : {total_frames}")
    print(f"  Duration   : {total_frames/fps:.1f}s\n")

    # ── Output video writer ──────────────────
    output_path = None
    writer = None
    if save_output:
        base, ext = os.path.splitext(video_path)
        output_path = base + "_analyzed.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))
        print(f"[Video] Saving annotated output → {output_path}\n")

    # ── Init modules ─────────────────────────
    print("Initializing AI modules…")
    os.makedirs(config.DATA_DIR, exist_ok=True)

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
    emotion_detector = EmotionDetector()
    gesture_detector = GestureDetector(
        model_path=config.POSE_MODEL,
        conf_threshold=config.GESTURE_CONFIDENCE_THRESHOLD,
        history_len=config.GESTURE_HISTORY_FRAMES,
    )
    context_engine = ContextEngine(
        vibe_window_fps=int(fps) // max(config.SKIP_FRAMES, 1),
        window_seconds=config.VIBE_WINDOW_SECONDS,
    )
    llm_engine = LLMEngine(
        api_key=config.LLM_API_KEY,
        model_name=config.LLM_MODEL_NAME,
        system_prompt=config.LLM_SYSTEM_PROMPT,
    )
    narrative_engine = NarrativeEngine(
        style=config.NARRATION_STYLE,
        summary_cooldown=config.NARRATION_SUMMARY_COOLDOWN,
        llm_engine=llm_engine if config.USE_LLM else None,
    )

    tts = None
    if use_tts:
        tts = TTSEngine(
            rate=config.TTS_RATE,
            volume=config.TTS_VOLUME,
            cooldown=config.TTS_COOLDOWN,
        )
        tts.start()

    # (Removed custom tracker as YOLOv8 handles it natively)

    print("All modules loaded. Starting analysis…")
    print("Controls: [Q] Quit  [Space] Pause  [S] Screenshot\n")

    frame_idx = 0
    paused = False
    last_narration = {}
    skip = max(config.SKIP_FRAMES, 1)
    face_align_counters = {}

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("\n[Video] End of file reached.")
                    break
                frame_idx += 1

            # ── Throttle to approx playback_speed ──
            delay = max(1, int((1000.0 / fps) / playback_speed))

            # ── Per-frame Analysis (every skip frames) ─────────
            if not paused and frame_idx % skip == 0:
                # 1. Detection AND Tracking (Native ByteTrack/BoT-SORT)
                # Returns: [{"person_id": T_ID, "bbox": (x1,y1,x2,y2), ...}]
                persons_detected = person_detector.detect(frame)
                
                # 2. Get Pose/Gestures matching those IDs
                track_ids = [p["person_id"] for p in persons_detected]
                gestures = gesture_detector.detect(frame, person_ids=track_ids)

                persons = []
                for i, person in enumerate(persons_detected):
                    bbox = person["bbox"]
                    tid = person["person_id"]

                    best_gesture, best_kpts = "Standing", None
                    for g in gestures:
                        if g["id"] == tid:
                            best_gesture = g["gesture"]
                            best_kpts = g["keypoints"]
                            break

                    # Proximity Gate: Skip heavy face/emotion if person is too far
                    x1, y1, x2, y2 = bbox
                    if (x2 - x1) < config.MIN_PROXIMITY_WIDTH:
                        aligned_face = None
                    else:
                        # Face Alignment Throttling: Only align every 5 frames
                        face_align_counters[tid] = face_align_counters.get(tid, 0) + 1
                        if face_align_counters[tid] % 5 == 0 or tid not in face_recognizer.identity_cache:
                            body_crop = person["cropped_body"]
                            aligned_face, _ = face_detector.detect_and_align(body_crop)
                        else:
                            aligned_face = None

                    # Recognize & Detect (Handles aligned_face=None via Cache)
                    identity = face_recognizer.recognize(aligned_face, person_id=tid)
                    emotion = emotion_detector.detect(aligned_face, person_id=tid)

                    if identity["name"] == "Unknown" and identity.get("embedding") is not None:
                        new_name = face_recognizer.auto_enroll(tid, identity["embedding"])
                        if new_name:
                            identity["name"] = new_name
                            print(f"  [Self-Learning] Auto-enrolled: {new_name}")

                    persons.append({
                        "person_id": tid,
                        "bbox": bbox,
                        "identity": identity,
                        "emotion": emotion,
                        "gesture": best_gesture,
                        "keypoints": best_kpts,
                    })

                # ── Context & Narration ─────────────────────────
                context_engine.update(persons)
                summary = context_engine.get_summary()
                interactions = context_engine.detect_interactions(persons)
                proactive = narrative_engine.generate_narration(persons, summary, interactions)
                if proactive and tts:
                    tts.speak(proactive)

                # ── Draw Annotations ───────────────────────────
                for p in persons:
                    name = p["identity"]["name"]
                    conf = p["identity"]["confidence"]
                    bbox = p["bbox"]
                    emo = p.get("emotion", {})
                    emo_label = emo.get("emotion", "")
                    emo_conf = emo.get("confidence", 0.0)
                    gesture = p.get("gesture", "Standing")
                    kpts = p.get("keypoints")
                    color = config.BOX_COLOR_KNOWN if name != "Unknown" else config.BOX_COLOR_UNKNOWN

                    # Individual TTS narration
                    if name != "Unknown" and tts:
                        now_t = time.time()
                        if now_t - last_narration.get(name, 0) > config.TTS_COOLDOWN:
                            msg = f"{name}"
                            if emo_label and emo_label != "Neutral" and emo_conf >= config.EMOTION_CONFIDENCE_THRESHOLD:
                                msg += f" looks {emo_label.lower()}"
                            if gesture and gesture != "Standing":
                                # Conjunction for natural speech
                                clean_gest = gesture.lower()
                                if "point" in clean_gest:
                                    clean_gest = clean_gest.replace("point", "pointing")
                                elif "handshake" in clean_gest:
                                    clean_gest = "offering a handshake"
                                
                                msg += f" and is {clean_gest}" if "looks" in msg else f" is {clean_gest}"
                            else:
                                msg += "." if "looks" in msg else " is here."
                            tts.speak(msg)
                            last_narration[name] = now_t

                    draw_person_box(frame, bbox, name, conf, color,
                                    config.FONT_SCALE, config.FONT_THICKNESS,
                                    emotion=emo_label, emotion_conf=emo_conf, gesture=gesture)
                    if kpts is not None:
                        draw_pose_keypoints(frame, kpts)

                draw_info_overlay(frame, fps, len(persons), vibe=summary["vibe"])
                draw_progress_bar(frame, frame_idx, total_frames)

                # ── Frame counter label ─────────────────────────
                ts = f"{int(frame_idx/fps//60):02d}:{int(frame_idx/fps%60):02d} / " \
                     f"{int(total_frames/fps//60):02d}:{int(total_frames/fps%60):02d}"
                cv2.putText(frame, ts, (10, frame_h - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

                if writer:
                    writer.write(frame)

            # ── Display ────────────────────────────────────────
            cv2.imshow("AI Analysis — Video Mode", frame)
            key = cv2.waitKey(delay) & 0xFF

            if key == ord("q") or key == ord("Q"):
                print("[Video] Quit by user.")
                break
            elif key == ord(" "):
                paused = not paused
                print("[Video] " + ("Paused." if paused else "Resumed."))
            elif key == ord("s") or key == ord("S"):
                shot_path = f"screenshot_{frame_idx}.png"
                cv2.imwrite(shot_path, frame)
                print(f"[Video] Screenshot saved → {shot_path}")

    except KeyboardInterrupt:
        print("\n[Video] Interrupted.")

    finally:
        cap.release()
        if writer:
            writer.release()
            print(f"\n[Video] Annotated video saved → {output_path}")
        if tts:
            tts.stop()
        cv2.destroyAllWindows()
        print("[Video] Done.")


def main():
    parser = argparse.ArgumentParser(description="AI Video File Analyzer")
    parser.add_argument("--video", type=str, default=None,
                        help="Path to the video file (opens file dialog if omitted)")
    parser.add_argument("--no-tts", action="store_true",
                        help="Disable text-to-speech narration")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed multiplier (e.g. 2.0 = 2x speed)")
    parser.add_argument("--no-save", action="store_true",
                        help="Do not save the annotated output video")
    args = parser.parse_args()

    video_path = args.video
    if not video_path:
        print("No video path provided. Opening file picker…")
        video_path = pick_video_file()

    if not video_path or not os.path.isfile(video_path):
        print(f"[ERROR] File not found or no file selected: {video_path}")
        sys.exit(1)

    run_analysis(
        video_path=video_path,
        use_tts=not args.no_tts,
        playback_speed=args.speed,
        save_output=not args.no_save,
    )


if __name__ == "__main__":
    main()
