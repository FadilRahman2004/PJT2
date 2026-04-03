# Multi-Modal Real-Time AI Assistant

This project is an advanced, real-time AI perception system designed to function as the "eyes and brain" of a robotic assistant or smart home hub. It analyzes live video (or pre-recorded files) to understand who is in the room, their emotions, their body language, and the overall "vibe" of the social context.

## 🌟 Key Features

1. **Persistent Person Tracking**: Tracks individuals across frames, even giving them temporary IDs ("Visitor_1") until they are identified.
2. **Autonomous Face Recognition**: Uses `InceptionResnetV1` (FaceNet). The system "learns" new faces dynamically. If it sees an unknown person consistently, it auto-enrolls them into its local, diversity-aware database.
3. **Triple-Ensemble Emotion Detection**: Uses specialized `EfficientNet` models from `hsemotion` (trained on VGAF, AFEW, and AffectNet) to provide robust, highly accurate facial emotion analysis.
4. **Hybrid Gesture & Posture Perception**: 
   - **Body Postures**: Uses `YOLOv8-pose` for large-scale analysis (Crossed Arms, Handshake offers).
   - **Fine-Grained Hands**: Uses `MediaPipe` Tasks for specific finger gestures (Thumbs Up, Open Palm).
   - **Head Tracking**: Temporal analysis of keypoints to detect Nodding (Yes) and Shaking Head (No).
5. **Generative Social Reasoning Engine**: Aggregates the visual data (who, what, how they feel) into a JSON context, which is passed to a Large Language Model (Gemini/OpenAI) to generate natural, conversational narrations of the room's vibe.

## 🛠️ Architecture Overview

The system runs on a highly optimized, multi-threaded pipeline:

- **Module 1**: `FrameCapture` (Video input queue)
- **Module 2**: `PersonDetector` (YOLOv8)
- **Module 3**: `FaceDetector` (Haar cascades with body cropping)
- **Module 4**: `FaceRecognizer` (FaceNet with auto-enrollment)
- **Module 5**: `EmotionDetector` (Triple-Ensemble `hsemotion`)
- **Module 6**: `GestureDetector` (Hybrid YOLO-Pose + MediaPipe)
- **Module 7**: `ContextEngine` (Social aggregation & Vibe calculation)
- **Module 8**: `NarrativeEngine` & `LLMEngine` (Proactive conversational AI)
- **Module 9**: `TTSEngine` (Text-to-Speech output via pyttsx3)

## 🚀 Setup & Installation

### Prerequisites
- Python 3.9+
- A working webcam (for live mode)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure the Brain (LLM)
Open `config.py` and modify the LLM settings. You will need a Google Gemini (or OpenAI) API Key to enable the generative social reasoning.
```python
USE_LLM = True
LLM_API_KEY = "your_actual_api_key_here"
```

### 3. Run the Assistant

**Live Webcam Mode:**
```bash
python main.py
```
*Controls:*
- `[R]` - Manually force registration of a new face.
- `[Q]` - Quit

**Video Analysis Mode:**
Analyze a pre-recorded video file instead of live feed.
```bash
python analyze_video.py --video path/to/video.mp4 --speed 2.0
```

## 📁 Project Structure

- `main.py` - Live webcam pipeline entry point.
- `analyze_video.py` - Video file analysis entry point.
- `config.py` - Central configuration for thresholds, models, and UI colors.
- `modules/` - Core AI and processing components.
- `utils/` - Helpers for drawing overlays and centroid tracking.
- `face_db/` - (Auto-generated) Local storage for facial embeddings.

---
*Note for Contributors: The next planned phase (Phase 7) is Voice Interaction (STT) to enable bi-directional conversation with the LLM.*
