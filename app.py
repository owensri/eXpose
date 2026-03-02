import streamlit as st
import os
import sys
import tempfile
import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from tensorflow.keras.models import load_model

# Import tools from our src folder
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
import config
from preprocessor import Preprocessor
from smoother import LandmarkSmoother

# ---------------------------------------------------------
# 1. Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Exercise Tracker",
    page_icon="🏋️‍♂️",
    layout="wide"
)

# ---------------------------------------------------------
# 2. Custom Drawing Function (12 Keypoints)
# ---------------------------------------------------------
TARGET_INDICES = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

CUSTOM_CONNECTIONS = [
    (11, 12),                 # Shoulders
    (11, 13), (13, 15),       # Left Arm
    (12, 14), (14, 16),       # Right Arm
    (11, 23), (12, 24),       # Torso
    (23, 24),                 # Hips
    (23, 25), (25, 27),       # Left Leg
    (24, 26), (26, 28)        # Right Leg
]

def draw_custom_landmarks(image, landmarks_list):
    """Draw only the 12 specific landmarks and connections"""
    h, w, _ = image.shape
    
    # 1. Draw Connections
    for connection in CUSTOM_CONNECTIONS:
        idx1, idx2 = connection
        
        lm1 = landmarks_list.landmark[idx1]
        lm2 = landmarks_list.landmark[idx2]
        
        if lm1.visibility > 0.5 and lm2.visibility > 0.5:
            pt1 = (int(lm1.x * w), int(lm1.y * h))
            pt2 = (int(lm2.x * w), int(lm2.y * h))
            cv2.line(image, pt1, pt2, (245, 117, 66), 3)

    # 2. Draw Landmarks (Circles)
    for idx in TARGET_INDICES:
        lm = landmarks_list.landmark[idx]
        if lm.visibility > 0.5:
            pt = (int(lm.x * w), int(lm.y * h))
            cv2.circle(image, pt, 6, (245, 66, 230), -1) 
            cv2.circle(image, pt, 8, (255, 255, 255), 2) 

# ---------------------------------------------------------
# 3. Main Function
# ---------------------------------------------------------
def main():
    st.title("🏋️‍♂️ AI Exercise Tracker")
    st.markdown("ระบบประเมินท่าออกกำลังกายด้วยปัญญาประดิษฐ์ (Real-time Pose Estimation)")
    st.markdown("---")

    # ==========================================
    # Sidebar
    # ==========================================
    with st.sidebar:
        st.header("⚙️ ตั้งค่าการทำงาน")
        
        st.subheader("1. เลือกสมอง AI (Model)")
        model_choice = st.selectbox(
            "โมเดลที่ต้องการใช้งาน:",
            ["Hybrid (CNN + LSTM) - แม่นยำสูงสุด", 
             "CNN - เน้นภาพนิ่ง", 
             "LSTM - เน้นความต่อเนื่อง"]
        )
        
        model_files = {
            "Hybrid (CNN + LSTM) - แม่นยำสูงสุด": "exercise_model_hybrid.keras",
            "CNN - เน้นภาพนิ่ง": "exercise_model_cnn.keras",
            "LSTM - เน้นความต่อเนื่อง": "exercise_model_lstm.keras"
        }
        selected_model_file = model_files[model_choice]
        
        # Absolute path for model loading
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, selected_model_file)

        st.markdown("---")

        st.subheader("2. อัปโหลดคลิปวิดีโอ")
        uploaded_file = st.file_uploader(
            "เลือกไฟล์วิดีโอ (MP4, MOV, AVI)", 
            type=['mp4', 'mov', 'avi']
        )

        st.markdown("---")
        
        start_button = st.button("🚀 เริ่มประมวลผลวิดีโอ", use_container_width=True, type="primary")

    # ==========================================
    # Main Area
    # ==========================================
    if uploaded_file is None:
        st.info("👈 กรุณาอัปโหลดคลิปวิดีโอจากแถบเมนูด้านซ้ายเพื่อเริ่มต้นใช้งานครับ")
    else:
        if not start_button:
            st.success(f"โหลดไฟล์ {uploaded_file.name} สำเร็จ! กดปุ่ม 'เริ่มประมวลผลวิดีโอ' ได้เลยครับ")
            st.video(uploaded_file)
            
        else:
            st.subheader(f"📊 ผลการวิเคราะห์แบบ Real-time (ใช้โมเดล: {model_choice.split('-')[0].strip()})")
            
            if not os.path.exists(model_path):
                st.error(f"❌ ไม่พบไฟล์โมเดล '{selected_model_file}' กรุณาตรวจสอบให้แน่ใจว่าได้นำไฟล์มาวางในโฟลเดอร์เดียวกับ app.py แล้ว")
                return

            # Load Model and Tools
            model = load_model(model_path)
            classes = config.CLASSES
            
            mp_pose = mp.solutions.pose
            pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
            
            preprocessor = Preprocessor()
            smoother = LandmarkSmoother()
            window_frames = deque(maxlen=config.SEQUENCE_LENGTH)

            # Placeholders for UI
            col1, col2 = st.columns(2)
            with col1:
                action_metric = st.empty() 
                action_metric.metric(label="Current Action (ท่าปัจจุบัน)", value="Buffering...")
            with col2:
                conf_metric = st.empty()   
                conf_metric.metric(label="Confidence (ความมั่นใจ)", value="0.0 %")
                
            st.markdown("---")
            video_placeholder = st.empty() 
            
            tfile = tempfile.NamedTemporaryFile(delete=False) 
            tfile.write(uploaded_file.read())
            
            cap = cv2.VideoCapture(tfile.name)
            
            current_action = "Buffering..."
            confidence = 0.0
            
            # [NEW] Counter for Optimization
            frame_counter = 0

            # ---------------------------------------------------------
            # Video Processing Loop
            # ---------------------------------------------------------
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_counter += 1

                # [OPTIMIZATION 1] Reduce resolution to 800x450 (16:9) to save CPU
                frame = cv2.resize(frame, (800, 450))
                
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_rgb.flags.writeable = False
                results = pose.process(image_rgb)
                image_rgb.flags.writeable = True
                
                if results.pose_landmarks:
                    draw_custom_landmarks(image_rgb, results.pose_landmarks)

                    lms = preprocessor.get_landmarks(results)
                    if lms:
                        lms = smoother.process(lms)
                        features = preprocessor.normalize(lms)
                        window_frames.append(features)

                    if len(window_frames) == config.SEQUENCE_LENGTH:
                        # [OPTIMIZATION 2] Predict only every 3 frames
                        if frame_counter % 3 == 0:
                            input_data = np.expand_dims(np.array(window_frames), axis=0)
                            predictions = model.predict(input_data, verbose=0)[0]
                            
                            best_class_idx = np.argmax(predictions)
                            
                            # Anti-Bias Logic
                            OTHER_THRESHOLD = 0.85 
                            if "other" in classes:
                                other_idx = classes.index("other")
                                if best_class_idx == other_idx and predictions[other_idx] < OTHER_THRESHOLD:
                                    temp_preds = predictions.copy()
                                    temp_preds[other_idx] = 0.0 
                                    best_class_idx = np.argmax(temp_preds)
                            
                            confidence = predictions[best_class_idx]
                            current_action = classes[best_class_idx]

                # [OPTIMIZATION 3] Update Streamlit UI only every 2 frames
                # Prevents WebSocket network flooding and reduces lag
                if frame_counter % 2 == 0:
                    if len(window_frames) < config.SEQUENCE_LENGTH:
                        action_metric.metric(label="Current Action (ท่าปัจจุบัน)", value=f"Buffering... {len(window_frames)}/{config.SEQUENCE_LENGTH}")
                        conf_metric.metric(label="Confidence (ความมั่นใจ)", value="-")
                    else:
                        action_metric.metric(label="Current Action (ท่าปัจจุบัน)", value=current_action.upper())
                        conf_metric.metric(label="Confidence (ความมั่นใจ)", value=f"{confidence*100:.1f} %")

                    video_placeholder.image(image_rgb, channels="RGB", use_container_width=True)

            cap.release()
            st.success("✅ ประมวลผลวิดีโอเสร็จสิ้น!")

if __name__ == "__main__":
    main()
