import streamlit as st
import os
import sys
import tempfile
import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from tensorflow.keras.models import load_model

# นำเข้าเครื่องมือจากโฟลเดอร์ src ของเรา
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
import config
from preprocessor import Preprocessor
from smoother import LandmarkSmoother

# ---------------------------------------------------------
# 1. ตั้งค่าหน้าเว็บ (Page Configuration)
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Exercise Tracker",
    page_icon="🏋️‍♂️",
    layout="wide"
)

# ---------------------------------------------------------
# 2. ฟังก์ชันเสริมสำหรับวาดเฉพาะ 12 จุด (Custom Drawing)
# ---------------------------------------------------------
# จุดที่ 11 ถึง 28 (ไหล่ถึงข้อเท้า)
TARGET_INDICES = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

# กำหนดเส้นเชื่อมกระดูก (Connections) สำหรับ 12 จุดนี้
CUSTOM_CONNECTIONS = [
    (11, 12),                 # ไหล่ซ้าย - ไหล่ขวา
    (11, 13), (13, 15),       # แขนซ้าย
    (12, 14), (14, 16),       # แขนขวา
    (11, 23), (12, 24),       # ลำตัวซ้าย-ขวา
    (23, 24),                 # สะโพกซ้าย - สะโพกขวา
    (23, 25), (25, 27),       # ขาซ้าย
    (24, 26), (26, 28)        # ขาขวา
]

def draw_custom_landmarks(image, landmarks_list):
    """ฟังก์ชันสำหรับวาดวงกลมและเส้นเชื่อมเฉพาะ 12 จุดสำคัญ"""
    h, w, _ = image.shape
    
    # 1. วาดเส้นเชื่อม (Connections)
    for connection in CUSTOM_CONNECTIONS:
        idx1, idx2 = connection
        
        lm1 = landmarks_list.landmark[idx1]
        lm2 = landmarks_list.landmark[idx2]
        
        # วาดเฉพาะจุดที่ AI มองเห็น (Visibility > 0.5)
        if lm1.visibility > 0.5 and lm2.visibility > 0.5:
            pt1 = (int(lm1.x * w), int(lm1.y * h))
            pt2 = (int(lm2.x * w), int(lm2.y * h))
            cv2.line(image, pt1, pt2, (245, 117, 66), 3) # สีส้ม-แดง

    # 2. วาดวงกลมตรงข้อต่อ (Landmarks)
    for idx in TARGET_INDICES:
        lm = landmarks_list.landmark[idx]
        if lm.visibility > 0.5:
            pt = (int(lm.x * w), int(lm.y * h))
            cv2.circle(image, pt, 6, (245, 66, 230), -1) # สีชมพูทึบ
            cv2.circle(image, pt, 8, (255, 255, 255), 2) # ขอบสีขาว

# ---------------------------------------------------------
# 3. ส่วนหลักของโปรแกรม (Main Function)
# ---------------------------------------------------------
def main():
    st.title("🏋️‍♂️ AI Exercise Tracker")
    st.markdown("ระบบประเมินท่าออกกำลังกายด้วยปัญญาประดิษฐ์ (Real-time Pose Estimation)")
    st.markdown("---")

    # ==========================================
    # ส่วน Sidebar (แถบเมนูด้านซ้าย)
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
        
        # [แก้ไข] สร้าง Absolute Path เพื่อให้หาไฟล์โมเดลเจอ 100% ไม่ว่ารันจากที่ไหน
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
    # ส่วน Main Area (พื้นที่แสดงผลหลัก)
    # ==========================================
    if uploaded_file is None:
        st.info("👈 กรุณาอัปโหลดคลิปวิดีโอจากแถบเมนูด้านซ้ายเพื่อเริ่มต้นใช้งานครับ")
    else:
        if not start_button:
            st.success(f"โหลดไฟล์ {uploaded_file.name} สำเร็จ! กดปุ่ม 'เริ่มประมวลผลวิดีโอ' ได้เลยครับ")
            st.video(uploaded_file)
            
        else:
            st.subheader(f"📊 ผลการวิเคราะห์แบบ Real-time (ใช้โมเดล: {model_choice.split('-')[0].strip()})")
            
            # เช็คว่ามีไฟล์โมเดลในเครื่องไหม (ใช้ Absolute Path)
            if not os.path.exists(model_path):
                st.error(f"❌ ไม่พบไฟล์โมเดล '{selected_model_file}' กรุณาตรวจสอบให้แน่ใจว่าได้นำไฟล์มาวางในโฟลเดอร์เดียวกับ app.py แล้ว")
                return

            # โหลด AI และเครื่องมือ
            model = load_model(model_path)
            classes = config.CLASSES
            
            mp_pose = mp.solutions.pose
            pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
            
            preprocessor = Preprocessor()
            smoother = LandmarkSmoother()
            window_frames = deque(maxlen=config.SEQUENCE_LENGTH)

            # ---------------------------------------------------------
            # สร้างพื้นที่สำหรับอัปเดตแบบ Real-time (Placeholders)
            # ---------------------------------------------------------
            col1, col2 = st.columns(2)
            with col1:
                action_metric = st.empty() # จองพื้นที่สำหรับตัวหนังสือ
                action_metric.metric(label="Current Action (ท่าปัจจุบัน)", value="Buffering...")
            with col2:
                conf_metric = st.empty()   # จองพื้นที่สำหรับตัวหนังสือ
                conf_metric.metric(label="Confidence (ความมั่นใจ)", value="0.0 %")
                
            st.markdown("---")
            video_placeholder = st.empty() # จองพื้นที่สำหรับแสดงวิดีโอ
            
            # สร้างไฟล์ชั่วคราวเพื่อรับวิดีโอที่อัปโหลดมาให้ OpenCV อ่านได้
            tfile = tempfile.NamedTemporaryFile(delete=False) 
            tfile.write(uploaded_file.read())
            
            cap = cv2.VideoCapture(tfile.name)
            
            current_action = "Buffering..."
            confidence = 0.0

            # ---------------------------------------------------------
            # ลูปประมวลผลวิดีโอทีละเฟรม
            # ---------------------------------------------------------
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # ย่อขนาดวิดีโอเพื่อไม่ให้กินทรัพยากรมากเกินไป (แนะนำที่ 720p 16:9)
                frame = cv2.resize(frame, (1280, 720))
                
                # แปลงสีให้ MediaPipe
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_rgb.flags.writeable = False
                results = pose.process(image_rgb)
                image_rgb.flags.writeable = True
                
                # ถ้าพบโครงกระดูก
                if results.pose_landmarks:
                    # เรียกใช้ฟังก์ชันวาดจุดเฉพาะ 12 จุด
                    draw_custom_landmarks(image_rgb, results.pose_landmarks)

                    # สกัดพิกัด -> ลดการสั่น -> ทำ Normalization
                    lms = preprocessor.get_landmarks(results)
                    if lms:
                        lms = smoother.process(lms)
                        features = preprocessor.normalize(lms)
                        window_frames.append(features)

                    # เมื่อเก็บข้อมูลครบตามกำหนด ให้ AI เริ่มทายผล
                    if len(window_frames) == config.SEQUENCE_LENGTH:
                        input_data = np.expand_dims(np.array(window_frames), axis=0)
                        predictions = model.predict(input_data, verbose=0)[0]
                        
                        best_class_idx = np.argmax(predictions)
                        
                        # ระบบลดความลำเอียง (Anti-Bias) สำหรับคลาส OTHER
                        OTHER_THRESHOLD = 0.85 
                        if "other" in classes:
                            other_idx = classes.index("other")
                            if best_class_idx == other_idx and predictions[other_idx] < OTHER_THRESHOLD:
                                temp_preds = predictions.copy()
                                temp_preds[other_idx] = 0.0 
                                best_class_idx = np.argmax(temp_preds)
                        
                        confidence = predictions[best_class_idx]
                        current_action = classes[best_class_idx]

                # อัปเดตตัวเลขบนหน้าจอ
                if len(window_frames) < config.SEQUENCE_LENGTH:
                    action_metric.metric(label="Current Action (ท่าปัจจุบัน)", value=f"Buffering... {len(window_frames)}/{config.SEQUENCE_LENGTH}")
                    conf_metric.metric(label="Confidence (ความมั่นใจ)", value="-")
                else:
                    action_metric.metric(label="Current Action (ท่าปัจจุบัน)", value=current_action.upper())
                    conf_metric.metric(label="Confidence (ความมั่นใจ)", value=f"{confidence*100:.1f} %")

                # ส่งภาพเฟรมปัจจุบันขึ้นไปโชว์บนหน้าเว็บ
                video_placeholder.image(image_rgb, channels="RGB", use_container_width=True)

            cap.release()
            st.success("✅ ประมวลผลวิดีโอเสร็จสิ้น!")

if __name__ == "__main__":
    main()