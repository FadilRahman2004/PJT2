"""
Central configuration for the Multi-Modal Real-Time AI Assistant.
All tunables live here — adjust as needed for your hardware.
"""

import os
from dotenv import load_dotenv
load_dotenv()  # Loads .env file if present (local dev / GPU server)

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
FACE_DB_PATH = os.path.join(DATA_DIR, "faces_db.json")

# ──────────────────────────────────────────────
# Frame Capture
# ──────────────────────────────────────────────
CAMERA_INDEX = 0                # Webcam device index
FRAME_WIDTH = 640               # Resize width
FRAME_HEIGHT = 480              # Resize height
TARGET_FPS = 20                 # Target capture FPS (lowered from 25 — reduces pipeline pressure)
FRAME_QUEUE_SIZE = 2            # Max frames buffered between threads (keep low to avoid stale-frame lag)

# ──────────────────────────────────────────────
# Person Detection (YOLOv8)
# ──────────────────────────────────────────────
YOLO_MODEL = "yolov8n.pt"      # Model variant (nano for speed)
YOLO_CONFIDENCE = 0.5           # Min detection confidence
YOLO_PERSON_CLASS = 0           # COCO class id for 'person'

# ──────────────────────────────────────────────
# Face Detection & Alignment (OpenCV Haar Cascade)
# ──────────────────────────────────────────────
FACE_ALIGN_SIZE = (160, 160)    # FaceNet expects 160×160
HAAR_SCALE_FACTOR = 1.1         # Cascade scale factor
HAAR_MIN_NEIGHBORS = 5          # Min neighbours (higher = fewer false positives)
HAAR_MIN_FACE_SIZE = (40, 40)   # Minimum face size to detect

# ──────────────────────────────────────────────
# Face Recognition (FaceNet via facenet-pytorch)
# ──────────────────────────────────────────────
FACENET_PRETRAINED = "vggface2"  # Pretrained dataset: 'vggface2' or 'casia-webface'
FACE_MATCH_THRESHOLD = 0.70      # Cosine similarity threshold for positive match
MAX_EMBEDDINGS_PER_PERSON = 50   # Cap stored embeddings per identity
AUTO_ENROLL_THRESHOLD = 25       # Frames to track unknown before auto-enrollment
FACE_CACHE_THRESHOLD = 0.85      # Strength to lock identity
FACE_CACHE_EXPIRY = 60           # Frames before re-checking identity
FACE_DATASET_DIR = "data/face_dataset"
FACE_DATASET_LIMIT = 50          # Max images per person in dataset
FACE_CAPTURE_INTERVAL = 30       # Capture an image every N frames of visibility

# ──────────────────────────────────────────────
# Emotion Detection Experts (Triple Ensemble)
EMOTION_EXPERTS = [
    'enet_b0_8_best_vgaf', # Video-based
    'enet_b0_8_best_afew', # Wild clips
    'enet_b0_8_va_mtl'     # High-precision static
]
EMOTION_CONFIDENCE_THRESHOLD = 0.35  # Slightly higher for ensemble
EMOTION_MIN_FACE_SIZE = 40           # Skip emotion if face is smaller than 40x40
EMOTION_ALPHA = 0.2                  # Temporal smoothing (higher = faster response)
EMOTION_SKIP_FRAMES = 8              # Only run neural model every N frames (higher = faster, was 5)
GESTURE_SKIP_FRAMES = 5              # Only run Holistic/ML if no motion detected or every N frames (higher = faster, was 3)
MIN_PROXIMITY_WIDTH = 80             # Min bbox width to run Face/Emotion

# ──────────────────────────────────────────────
# Gesture & Pose Detection (YOLOv8-pose)
# ──────────────────────────────────────────────
POSE_MODEL = "yolov8n-pose.pt"   # Lightweight YOLOv8-pose model
GESTURE_CONFIDENCE_THRESHOLD = 0.5  # Min confidence for pose keypoints
GESTURE_HISTORY_FRAMES  = 25       # Extended history for movement analysis
POSE_BOX_SCALE          = 1.2      # Scale up body box for pose analysis
GESTURE_MIN_MOTION_THRESHOLD = 8.0 # Min pixels a keypoint must move for dynamic gestures
ONE_EURO_MIN_CUTOFF     = 1.0      # Filter param: lower = less jitter but more lag
ONE_EURO_BETA           = 0.01     # Filter param: higher = less lag at high speeds
HOLISTIC_COMPLEXITY     = 0        # MediaPipe Holistic: 0=fast, 1=balanced, 2=accurate — use 0 when running full pipeline
# Normalised-space thresholds (0.0–1.0, resolution-independent)
NOD_NORM_THRESHOLD   = 0.018       # Nose Y std-dev for nodding (lowered from 0.025)
SHAKE_NORM_THRESHOLD = 0.022       # Nose X std-dev for head shaking

# ──────────────────────────────────────────────
# Room Vibe & Narrative Engine
# ──────────────────────────────────────────────
VIBE_WINDOW_SECONDS = 5.0       # Seconds of history for vibe calculation

# TTS Settings (Mouth)
TTS_RATE = 160
TTS_VOLUME = 0.9
TTS_COOLDOWN = 4.0

# STT Settings (Ears)
USE_STT = True                  # Toggle microphone listening
STT_WAKE_WORD = "assistant"     # The word that triggers the LLM query
STT_TIMEOUT = 5                 # Seconds to listen for a command after waking
STT_PHRASE_LIMIT = 10           # Max seconds of speech to record per query

# Narration / Concept Settings
NARRATION_STYLE = "proactive"
   # 'proactive' or 'reactive'
NARRATION_SUMMARY_COOLDOWN = 45.0  # Seconds between room mood summaries

# LLM Integration (Brain)
USE_LLM = True                 # Toggle LLM-based narration
LLM_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_API_KEY_HERE")    # Set your Google Gemini API Key in a .env file
LLM_MODEL_NAME = "gemini-2.0-flash"
LLM_COOLDOWN = 20.0            # Seconds between LLM "Brain" evaluations
LLM_URGENCY_THRESHOLD = 8       # Importance score (1-10) that breaks narrative silence
LLM_SYSTEM_PROMPT = """
You are the central "Brain" of an advanced AI home assistant for the visually impaired.
You receive a JSON payload containing the current social context: the overall vibe, people in the room, their emotions, gestures, and the time since you last spoke.

### CONVERSATIONAL RULES:
1. **BE HUMAN**: Do not call people by their full names repeatedly (e.g., say "John" instead of "John Doe"). 
2. **PRONOUNS**: After identifying someone once, use pronouns (he/she/they) or "The person" instead of their name.
3. **SOCIAL SILENCE**: Do not narrate every tiny change. Only speak if:
   - Someone new enters.
   - Someone waves or performs a notable gesture (like 'Handshake' or 'Pointing').
   - The overall vibe of the room changes significantly.
   - You haven't spoken in a long time (cooldown over) AND it is a polite moment to summarize.
4. **NO REPETITION**: If the situation hasn't changed since your last message, stay quiet (should_speak = false).

RESPOND ONLY IN VALID JSON FORMAT MATCHING THIS SCHEMA:
{
  "should_speak": boolean, // True if the context warrants a verbal comment, False if you should stay quiet.
  "importance": integer,   // 1 to 10. 10=Action needed (Stop!), 8=Direct Greeting, 4=Context change, 1=Boring.
  "message": string,       // 1 concise, natural sentence. No robot-speak. Use pronouns!
  "reasoning": string      // Brief internal thought on why you decided to speak or stay quiet.
}
"""

# ──────────────────────────────────────────────
# Unknown Person Registration
# ──────────────────────────────────────────────
UNKNOWN_PERSIST_FRAMES = 30     # Frames an unknown must persist before prompt
REGISTRATION_CAPTURE_COUNT = 10 # Frames captured during registration

# ──────────────────────────────────────────────
# TTS (pyttsx3)
# ──────────────────────────────────────────────
TTS_RATE = 175                  # Words per minute
TTS_VOLUME = 0.9                # 0.0 – 1.0
TTS_COOLDOWN = 5.0              # Seconds between repeated narrations

# ──────────────────────────────────────────────
# Performance / Threading
# ──────────────────────────────────────────────
SKIP_FRAMES = 2                 # Process every Nth frame (1 = all, 2 = every 2nd) — raise to 3 if still slow

# ──────────────────────────────────────────────
# Display
# ──────────────────────────────────────────────
SHOW_WINDOW = True              # Show OpenCV preview window
WINDOW_NAME = "AI Assistant"
BOX_COLOR_KNOWN = (0, 255, 0)   # Green  (BGR)
BOX_COLOR_UNKNOWN = (0, 0, 255) # Red    (BGR)
FONT_SCALE = 0.6
FONT_THICKNESS = 2
