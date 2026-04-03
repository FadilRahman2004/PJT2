"""
Test Script: Person Detection & Tracking
Verifies YOLOv8 Native Tracking (ByteTrack/BoT-SORT).
"""
import cv2
import config
from modules.person_detector import PersonDetector
from utils.drawing import draw_person_box

def main():
    print("[Test] Initializing Person Detector...")
    detector = PersonDetector(
        model_path=config.PERSON_MODEL,
        confidence=config.PERSON_CONFIDENCE_THRESHOLD
    )
    
    cap = cv2.VideoCapture(0)
    print("Controls: [Q] Quit")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # 1. Detect and Track
        persons = detector.detect(frame)
        
        # 2. Visualize
        for p in persons:
            bbox = p["bbox"]
            tid = p["person_id"]
            draw_person_box(frame, bbox, f"ID: {tid}", (0, 255, 0))
            
        cv2.imshow("Person Tracking Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
