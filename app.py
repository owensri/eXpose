import streamlit as st
import os
import sys
import tempfile
import cv2
import numpy as np
import mediapipe as mp
import time
import subprocess
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
# 2. Helper Functions (วาดจุด และ คำนวณมุม)
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
    """วาดเส้นและจุดเฉพาะ 12 จุดสำคัญ"""
    h, w, _ = image.shape
    for connection in CUSTOM_CONNECTIONS:
        idx1, idx2 = connection
        lm1 = landmarks_list.landmark[idx1]
        lm2 = landmarks_list.landmark[idx2]
        if lm1.visibility > 0.5 and lm2.visibility > 0.5:
            pt1 = (int(lm1.x * w), int(lm1.y * h))
            pt2 = (int(lm2.x * w), int(lm2.y * h))
            cv2.line(image, pt1, pt2, (245, 117, 66), 3)

    for idx in TARGET_INDICES:
        lm = landmarks_list.landmark[idx]
        if lm.visibility > 0.5:
            pt = (int(lm.x * w), int(lm.y * h))
            cv2.circle(image, pt, 6, (245, 66, 230), -1) 
            cv2.circle(image, pt, 8, (255, 255, 255), 2) 

def calculate_angle(a, b, c):
    """คำนวณมุมองศาระหว่าง 3 จุด (เช่น ไหล่-ศอก-ข้อมือ)"""
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

# ---------------------------------------------------------
# 3. Main Function
# ---------------------------------------------------------
def main():
    st.title("🏋️‍♂️ AI Exercise Tracker")
    st.markdown("ระบบประเมินท่าออกกำลังกายและนับจำนวนครั้งด้วย AI")
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
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, selected_model_file)

        st.markdown("---")

        st.subheader("2. โหมดการแสดงผล")
        display_mode = st.radio(
            "เลือกรูปแบบการประมวลผลวิดีโอ:",
            ["🚀 Fast Mode (ปิดพรีวิว - ประมวลผลเร็วที่สุด)", 
             "👁️ Live Preview (แสดงภาพขณะวิเคราะห์ - ใช้เวลามากขึ้น)"]
        )
        is_fast_mode = "Fast Mode" in display_mode

        st.markdown("---")

        st.subheader("3. อัปโหลดคลิปวิดีโอ")
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
        st.info("👈 กรุณาอัปโหลดคลิปวิดีโอและเลือกการตั้งค่าจากแถบเมนูด้านซ้ายเพื่อเริ่มต้นใช้งานครับ")
    else:
        if not start_button:
            st.success(f"โหลดไฟล์ {uploaded_file.name} สำเร็จ! กดปุ่ม 'เริ่มประมวลผลวิดีโอ' ได้เลยครับ")
            st.video(uploaded_file)
            
        else:
            st.subheader(f"📊 กำลังประมวลผล... (ใช้โมเดล: {model_choice.split('-')[0].strip()})")
            
            if not os.path.exists(model_path):
                st.error(f"❌ ไม่พบไฟล์โมเดล '{selected_model_file}' กรุณาตรวจสอบให้แน่ใจว่าได้นำไฟล์มาวางในโฟลเดอร์เดียวกับ app.py แล้ว")
                return

            overall_start_time = time.time()

            model = load_model(model_path)
            classes = config.CLASSES
            
            mp_pose = mp.solutions.pose
            pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
            
            preprocessor = Preprocessor()
            smoother = LandmarkSmoother()
            window_frames = deque(maxlen=config.SEQUENCE_LENGTH)

            progress_bar = st.progress(0)
            status_text = st.empty()
            video_placeholder = st.empty() 
            
            tfile = tempfile.NamedTemporaryFile(delete=False) 
            tfile.write(uploaded_file.read())
            cap = cv2.VideoCapture(tfile.name)
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0 or np.isnan(fps):
                fps = 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            temp_out_mp4 = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
            final_out_mp4 = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(temp_out_mp4, fourcc, fps, (800, 450))
            
            current_action = "Buffering..."
            confidence = 0.0
            frame_counter = 0

            # --- ตัวแปรสำหรับนับรอบ (Repetition Counter) ---
            counter = 0 
            stage = None # สถานะปัจจุบัน (up หรือ down)

            if is_fast_mode:
                video_placeholder.info("⏳ กำลังวิเคราะห์และนับจำนวนครั้งแบบ **Fast Processing**...")

            # ---------------------------------------------------------
            # Video Processing Loop
            # ---------------------------------------------------------
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_counter += 1
                frame = cv2.resize(frame, (800, 450))
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                image_rgb.flags.writeable = False
                results = pose.process(image_rgb)
                image_rgb.flags.writeable = True
                
                if results.pose_landmarks:
                    draw_custom_landmarks(image_rgb, results.pose_landmarks)
                    landmarks = results.pose_landmarks.landmark

                    lms = preprocessor.get_landmarks(results)
                    if lms:
                        lms = smoother.process(lms)
                        features = preprocessor.normalize(lms)
                        window_frames.append(features)

                    # 1. ทายผลว่าเป็นท่าอะไร (AI Prediction)
                    if len(window_frames) == config.SEQUENCE_LENGTH:
                        if frame_counter % 3 == 0:
                            input_data = np.expand_dims(np.array(window_frames), axis=0)
                            predictions = model.predict(input_data, verbose=0)[0]
                            best_class_idx = np.argmax(predictions)
                            
                            OTHER_THRESHOLD = 0.85 
                            if "other" in classes:
                                other_idx = classes.index("other")
                                if best_class_idx == other_idx and predictions[other_idx] < OTHER_THRESHOLD:
                                    temp_preds = predictions.copy()
                                    temp_preds[other_idx] = 0.0 
                                    best_class_idx = np.argmax(temp_preds)
                            
                            confidence = predictions[best_class_idx]
                            current_action = classes[best_class_idx]

                    # 2. ระบบนับจำนวนครั้ง (Repetition Logic)
                    try:
                        # เช็คท่า Squat
                        if current_action == 'squat':
                            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                            
                            angle = calculate_angle(hip, knee, ankle)
                            
                            if angle > 160:
                                stage = "up"
                            if angle < 100 and stage == 'up':
                                stage = "down"
                                counter += 1

                        # เช็คท่า Pushup (วิดพื้น)
                        elif current_action == 'pushup':
                            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                            
                            angle = calculate_angle(shoulder, elbow, wrist)
                            
                            if angle > 160:
                                stage = "up"
                            if angle < 90 and stage == 'up':
                                stage = "down"
                                counter += 1
                                
                    except:
                        pass # เผื่อกรณีมองไม่เห็นจุดข้อต่อ ระบบจะไม่พัง

                # --- 3. แสดงผล UI บนวิดีโอ ---
                
                # แถบบอกท่าทางด้านซ้ายบน
                text_color = (0, 255, 0) if confidence > 0.8 else (255, 165, 0)
                if len(window_frames) < config.SEQUENCE_LENGTH:
                    display_text = f"Buffering... {len(window_frames)}/{config.SEQUENCE_LENGTH}"
                    text_color = (255, 255, 0)
                else:
                    display_text = f"Action: {current_action.upper()} ({confidence*100:.1f}%)"

                cv2.rectangle(image_rgb, (0, 0), (800, 50), (0, 0, 0), -1)
                cv2.putText(image_rgb, display_text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2, cv2.LINE_AA)

                # กล่องนับจำนวนครั้ง (Reps Counter) ด้านขวาบน
                cv2.rectangle(image_rgb, (600, 0), (800, 100), (245, 117, 16), -1)
                cv2.putText(image_rgb, 'REPS', (615, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2, cv2.LINE_AA)
                cv2.putText(image_rgb, str(counter), (615, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3, cv2.LINE_AA)
                
                cv2.putText(image_rgb, 'STAGE', (700, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2, cv2.LINE_AA)
                cv2.putText(image_rgb, str(stage) if stage else '-', (700, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2, cv2.LINE_AA)

                # บันทึกเฟรมที่วาดเสร็จแล้ว
                writer.write(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))

                if is_fast_mode:
                    if frame_counter % 15 == 0:
                        if total_frames > 0:
                            progress = min(frame_counter / total_frames, 1.0)
                            progress_bar.progress(progress)
                            status_text.text(f"🚀 กำลังรันโมเดลและนับจำนวน (Fast Mode)... {int(progress * 100)}%")
                else:
                    if frame_counter % 5 == 0:
                        video_placeholder.image(image_rgb, channels="RGB", use_container_width=True)
                        if total_frames > 0:
                            progress = min(frame_counter / total_frames, 1.0)
                            progress_bar.progress(progress)
                            status_text.text(f"👁️ กำลังวิเคราะห์และนับจำนวน (Live Preview)... {int(progress * 100)}%")

            cap.release()
            writer.release()
            
            # ---------------------------------------------------------
            # เข้ารหัสวิดีโอ
            # ---------------------------------------------------------
            status_text.text("กำลังเตรียมวิดีโอผลลัพธ์แบบสมูท 100% (Finalizing Video)...")
            cmd = ['ffmpeg', '-y', '-i', temp_out_mp4, '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', '-preset', 'fast', final_out_mp4]
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception as e:
                st.warning("⚠️ แปลงวิดีโอไม่สำเร็จ ระบบจะพยายามแสดงผลด้วยไฟล์ต้นฉบับ")
                final_out_mp4 = temp_out_mp4

            overall_end_time = time.time()
            total_duration = overall_end_time - overall_start_time

            video_placeholder.empty()
            progress_bar.empty()
            status_text.empty()
            
            st.success("✅ ประมวลผลเสร็จสิ้น! ดูผลลัพธ์ด้านล่างได้เลยครับ")
            
            # แสดง Metrics สรุปผลสวยๆ 3 กล่อง
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🔥 จำนวนครั้งที่ทำได้ (Reps)", f"{counter} ครั้ง")
            with col2:
                st.metric("⏱️ เวลาประมวลผลรวม", f"{total_duration:.2f} วินาที")
            with col3:
                st.metric("⚡ ความเร็วเฉลี่ย", f"{frame_counter / total_duration:.1f} FPS")

            st.video(final_out_mp4)

if __name__ == "__main__":
    main()
