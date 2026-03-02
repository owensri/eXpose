import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from collections import deque
import os

# นำเข้าไฟล์ตั้งค่าและตัวจัดการข้อมูลของเรา
import config
from preprocessor import Preprocessor
from smoother import LandmarkSmoother

def main():
    print("=========================================================")
    print(" 🎬 ระบบทดสอบ AI ประเมินท่าออกกำลังกาย (Anti-Bias Mode)")
    print("=========================================================\n")

    # ==========================================
    # 1. ตั้งค่าวิดีโอ และ โมเดลที่ต้องการทดสอบ
    # ==========================================
    VIDEO_PATH = "/Users/owensri/Downloads/test_001.mp4"
    MODEL_PATH = "exercise_model_hybrid.keras"
    
    if not os.path.exists(VIDEO_PATH):
        print(f"[ERROR] ไม่พบไฟล์วิดีโอที่: {VIDEO_PATH}")
        return

    print(f"[INFO] กำลังโหลดสมอง AI ({MODEL_PATH})...")
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] ไม่พบไฟล์โมเดล {MODEL_PATH}")
        return
        
    model = load_model(MODEL_PATH)
    classes = config.CLASSES

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    preprocessor = Preprocessor()
    smoother = LandmarkSmoother()

    window_frames = deque(maxlen=config.SEQUENCE_LENGTH)

    # 🔥 เพิ่ม buffer สำหรับทำ Temporal Voting
    prediction_buffer = deque(maxlen=5)

    cap = cv2.VideoCapture(VIDEO_PATH)
    current_action = "Buffering..."
    confidence = 0.0
    frame_counter = 0 

    print(f"[INFO] เริ่มประมวลผลคลิป: {VIDEO_PATH}")
    print("[INFO] กด 'q' เพื่อออก\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_counter += 1
        frame = cv2.resize(frame, (1280, 720))
        
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = pose.process(image_rgb)
        image_rgb.flags.writeable = True
        
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )

            lms = preprocessor.get_landmarks(results)
            if lms:
                lms = smoother.process(lms)
                features = preprocessor.normalize(lms)
                window_frames.append(features)

            if len(window_frames) == config.SEQUENCE_LENGTH:
                input_data = np.expand_dims(np.array(window_frames), axis=0)
                predictions = model.predict(input_data, verbose=0)[0]
                
                best_class_idx = np.argmax(predictions)

                # ==========================================
                # Anti-Bias Other
                # ==========================================
                OTHER_THRESHOLD = 0.85
                
                if "other" in classes:
                    other_idx = classes.index("other")
                    
                    if best_class_idx == other_idx and predictions[other_idx] < OTHER_THRESHOLD:
                        temp_preds = predictions.copy()
                        temp_preds[other_idx] = 0.0 
                        best_class_idx = np.argmax(temp_preds)

                # ==========================================
                # 🔥 Temporal Voting (Majority 5 windows)
                # ==========================================
                prediction_buffer.append(best_class_idx)

                if len(prediction_buffer) == prediction_buffer.maxlen:
                    voted_class_idx = max(set(prediction_buffer), key=prediction_buffer.count)
                else:
                    voted_class_idx = best_class_idx

                confidence = predictions[voted_class_idx]
                current_action = classes[voted_class_idx]

                if frame_counter % 5 == 0:
                    prob_str = " | ".join(
                        [f"{c.upper()}: {p*100:>5.1f}%" for c, p in zip(classes, predictions)]
                    )
                    print(f"Frame {frame_counter:04d} -> [ {prob_str} ] ==> Winner: {current_action.upper()}")

        # ==========================================
        # แสดงผล
        # ==========================================
        text_color = (0, 255, 0) if confidence > 0.8 else (0, 165, 255)

        if len(window_frames) < config.SEQUENCE_LENGTH:
            display_text = f"Buffering... {len(window_frames)}/{config.SEQUENCE_LENGTH}"
            text_color = (0, 255, 255)
        else:
            display_text = f"Action: {current_action.upper()} ({confidence*100:.1f}%)"

        cv2.rectangle(frame, (0, 0), (1280, 60), (0, 0, 0), -1)
        cv2.putText(frame, display_text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    text_color, 2, cv2.LINE_AA)
        
        cv2.imshow('AI Exercise Evaluation - 720p', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    print("\n[INFO] ปิดการทำงานเสร็จสิ้น")

if __name__ == "__main__":
    main()