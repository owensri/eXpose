import streamlit as st
import os
import sys
import tempfile
import cv2
import numpy as np
import mediapipe as mp
import time
import subprocess
import urllib.request
import pandas as pd
import altair as alt
from collections import deque
from tensorflow.keras.models import load_model
from PIL import Image, ImageDraw, ImageFont

# Import local modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
import config
from preprocessor import Preprocessor
from smoother import LandmarkSmoother
from rep_counter import RepCounter

# ---------------------------------------------------------
# Page Configuration & CSS
# ---------------------------------------------------------
st.set_page_config(page_title="AI Exercise Tracker", page_icon=None, layout="wide")

# CSS เฉพาะการตกแต่งความสวยงามของตัวหนังสือและกล่องสรุปผล (ไม่มีการซ่อน UI แล้ว)
st.markdown("""
    <style>
    .main { padding-top: 2rem; }
    h1 { font-weight: 600; letter-spacing: -0.5px; }
    h2, h3 { font-weight: 500; }
    div[data-testid="stMetric"] { background-color: #111111; padding: 20px; border-radius: 12px; text-align: center; }
    div[data-testid="stMetricValue"] { font-size: 28px; font-weight: 600; }
    div[data-testid="stMetricLabel"] { font-size: 14px; text-transform: uppercase; letter-spacing: 1px; }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Constants & Helpers
# ---------------------------------------------------------
TARGET_INDICES = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

CUSTOM_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), 
    (11, 23), (12, 24), (23, 24), 
    (23, 25), (25, 27), (24, 26), (26, 28)
]

OTHER_THRESHOLD = 0.85
VOTING_WINDOW = 5

def draw_custom_landmarks(image, landmarks_list):
    h, w, _ = image.shape
    for connection in CUSTOM_CONNECTIONS:
        idx1, idx2 = connection
        lm1, lm2 = landmarks_list.landmark[idx1], landmarks_list.landmark[idx2]
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

@st.cache_resource
def load_model_cached(path):
    return load_model(path, compile=False)

@st.cache_resource
def get_thai_font(size):
    font_name = "Sarabun-Regular.ttf"
    font_url = "https://github.com/google/fonts/raw/main/ofl/sarabun/Sarabun-Regular.ttf"
    
    if not os.path.exists(font_name):
        try:
            urllib.request.urlretrieve(font_url, font_name)
        except Exception as e:
            pass
            
    font_paths = [
        font_name, 
        "tahoma.ttf", 
        "/System/Library/Fonts/Supplemental/Tahoma.ttf", 
        "/System/Library/Fonts/Thonburi.ttc", 
        "C:\\Windows\\Fonts\\tahoma.ttf"
    ]
    
    for path in font_paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except:
                continue
    return ImageFont.load_default()

# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------
def main():
    st.markdown("<h1 style='text-align: center;'>AI Exercise Analysis System</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #888888;'>ระบบวิเคราะห์ท่าออกกำลังกายและนับจำนวนครั้งอย่างแม่นยำ</p><br>", unsafe_allow_html=True)

    with st.sidebar:
        st.header("⚙️ การตั้งค่าระบบ")
        model_choice = st.selectbox("เลือกโมเดล (Model):", ["LSTM + CNN", "CNN", "LSTM"])
        
        model_files = {
            "LSTM + CNN": "exercise_model_hybrid.keras",
            "CNN": "exercise_model_cnn.keras",
            "LSTM": "exercise_model_lstm.keras"
        }
        selected_model_file = model_files[model_choice]
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), selected_model_file)

        st.markdown("---")
        
        diff_display = st.selectbox("ความเข้มงวด (Difficulty Level):", 
                                    ["Beginner", "Advanced"], 
                                    index=0)
        
        difficulty_level = "beginner" if "Beginner" in diff_display else "advanced"

        st.markdown("---")
        uploaded_file = st.file_uploader("อัปโหลดวิดีโอ (Video File)", type=['mp4', 'mov', 'avi'])
        st.markdown("---")
        start_button = st.button("เริ่มวิเคราะห์ (Start Processing)", use_container_width=True, type="primary")

    if uploaded_file is None:
        st.info("กรุณาอัปโหลดวิดีโอและตั้งค่าระบบที่เมนูด้านซ้ายเพื่อเริ่มต้น")
    else:
        if not start_button:
            st.success(f"อัปโหลดไฟล์ '{uploaded_file.name}' สำเร็จ! กด 'เริ่มวิเคราะห์' ได้เลย")
            st.video(uploaded_file)
        else:
            st.subheader(f"กำลังประมวลผล (ความเข้มงวด: {difficulty_level.capitalize()})")
            
            if not os.path.exists(model_path):
                st.error(f"ไม่พบไฟล์โมเดล '{selected_model_file}'")
                return

            overall_start_time = time.time()
            model = load_model_cached(model_path)
            
            font_large = get_thai_font(28)
            font_medium = get_thai_font(22)

            classes = config.CLASSES
            mp_pose = mp.solutions.pose
            pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
            
            preprocessor = Preprocessor()
            smoother = LandmarkSmoother()
            rep_counter = RepCounter(difficulty=difficulty_level)
            
            window_frames = deque(maxlen=config.SEQUENCE_LENGTH)
            prediction_buffer = deque(maxlen=VOTING_WINDOW)

            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.info("กำลังประมวลผล (Fast Mode)... อาจใช้เวลาสักครู่ กรุณารอจนกว่าแถบสถานะจะเต็ม")
            
            tfile = tempfile.NamedTemporaryFile(delete=False) 
            tfile.write(uploaded_file.read())
            cap = cv2.VideoCapture(tfile.name)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            temp_out_mp4 = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
            final_out_mp4 = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
            
            writer = cv2.VideoWriter(temp_out_mp4, cv2.VideoWriter_fourcc(*'mp4v'), fps, (800, 550))
            
            current_action = "กำลังวิเคราะห์..."
            confidence = 0.0
            frame_counter = 0

            counters = {'pushup': 0, 'squat': 0, 'lunge': 0}
            stages = {'pushup': None, 'squat': None, 'lunge': None}
            feedback_msg = "รอภาพ..."
            fb_color = (255, 255, 255)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                    
                frame_counter += 1
                frame = cv2.resize(frame, (800, 450))
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                image_rgb.flags.writeable = False
                results = pose.process(image_rgb)
                image_rgb.flags.writeable = True
                
                angle_data = None
                
                if results.pose_landmarks:
                    draw_custom_landmarks(image_rgb, results.pose_landmarks)
                    landmarks = results.pose_landmarks.landmark
                    lms = preprocessor.get_landmarks(results)
                    
                    if lms:
                        lms = smoother.process(lms)
                        features = preprocessor.normalize(lms)
                        window_frames.append(features)

                    if len(window_frames) == config.SEQUENCE_LENGTH:
                        if frame_counter % 3 == 0:
                            input_data = np.expand_dims(np.array(window_frames), axis=0)
                            predictions = model.predict(input_data, verbose=0)[0]
                            best_idx = np.argmax(predictions)

                            if "other" in classes:
                                other_idx = classes.index("other")
                                if best_idx == other_idx and predictions[other_idx] < OTHER_THRESHOLD:
                                    temp_preds = predictions.copy()
                                    temp_preds[other_idx] = 0.0
                                    best_idx = np.argmax(temp_preds)

                            prediction_buffer.append(best_idx)
                            voted_idx = max(set(prediction_buffer), key=prediction_buffer.count) if len(prediction_buffer) == prediction_buffer.maxlen else best_idx
                            
                            confidence = predictions[voted_idx]
                            current_action = classes[voted_idx]

                    try:
                        counters, stages, feedback_msg, fb_color, angle_data = rep_counter.process(current_action, confidence, landmarks, mp_pose)
                    except Exception:
                        pass

                if angle_data:
                    h, w, _ = image_rgb.shape
                    
                    p_lm = angle_data['primary']['landmark']
                    p_angle = angle_data['primary']['angle']
                    px, py = int(p_lm.x * w), int(p_lm.y * h)
                    cv2.putText(image_rgb, f"{int(p_angle)}", (px + 15, py), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
                    cv2.putText(image_rgb, f"{int(p_angle)}", (px + 15, py), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
                    
                    s_lm = angle_data['secondary']['landmark']
                    # 📌 แก้ไขจาก ['landmark'] เป็น ['angle'] แล้วที่บรรทัดนี้
                    s_angle = angle_data['secondary']['angle']
                    sx, sy = int(s_lm.x * w), int(s_lm.y * h)
                    cv2.putText(image_rgb, f"{int(s_angle)}", (sx + 15, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
                    cv2.putText(image_rgb, f"{int(s_angle)}", (sx + 15, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 255), 2, cv2.LINE_AA)

                canvas = np.zeros((550, 800, 3), dtype=np.uint8)
                canvas[0:450, 0:800] = image_rgb
                
                cv2.rectangle(canvas, (0, 0), (800, 45), (0, 0, 0), -1)
                cv2.rectangle(canvas, (620, 0), (800, 110), (40, 40, 40), -1)

                pil_img = Image.fromarray(canvas)
                draw = ImageDraw.Draw(pil_img)

                current_stage = stages.get(current_action) or "-"
                action_color = (0, 255, 0) if confidence > 0.8 else (255, 165, 0)
                
                if len(window_frames) < config.SEQUENCE_LENGTH:
                    display_text = f"กำลังเก็บข้อมูล... {len(window_frames)}/{config.SEQUENCE_LENGTH}"
                    action_color = (255, 255, 0)
                else:
                    display_text = f"ท่าที่พบ: {current_action.upper()} ({confidence*100:.1f}%) | สถานะ: {current_stage.upper()}"

                draw.text((25, 12), display_text, font=font_medium, fill=action_color)
                draw.text((25, 465), f"ข้อเสนอแนะ: {feedback_msg}", font=font_large, fill=fb_color)

                draw.text((630, 10), f"PUSHUP: {counters['pushup']}", font=font_medium, fill=(255, 165, 0) if current_action == 'pushup' else (255,255,255))
                draw.text((630, 45), f"SQUAT : {counters['squat']}", font=font_medium, fill=(255, 165, 0) if current_action == 'squat' else (255,255,255))
                draw.text((630, 80), f"LUNGE : {counters['lunge']}", font=font_medium, fill=(255, 165, 0) if current_action == 'lunge' else (255,255,255))

                final_frame = np.array(pil_img)
                writer.write(cv2.cvtColor(final_frame, cv2.COLOR_RGB2BGR))

                if frame_counter % 15 == 0 and total_frames > 0:
                    progress_bar.progress(min(frame_counter / total_frames, 1.0))
                    status_text.info(f"กำลังประมวลผล (Fast Mode)... {int(min(frame_counter / total_frames, 1.0) * 100)}%")

            cap.release()
            writer.release()
            
            status_text.info("กำลังเตรียมวิดีโอผลลัพธ์ (Finalizing video)...")
            try:
                # ยังคงใส่ -an ไว้ เพื่อให้วิดีโอไม่มีเสียงมาตั้งแต่ต้นทาง (กันไว้ดีกว่าแก้)
                subprocess.run(['ffmpeg', '-y', '-i', temp_out_mp4, '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', '-preset', 'fast', '-an', final_out_mp4], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                final_out_mp4 = temp_out_mp4

            progress_bar.empty()
            status_text.empty()

            st.success("ประมวลผลเสร็จสมบูรณ์!")
            st.markdown("<h2 style='text-align:center;'>สรุปจำนวนครั้ง (Repetition Dashboard)</h2>", unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)
            col1.metric("Pushup", f"{counters['pushup']}")
            col2.metric("Squat", f"{counters['squat']}")
            col3.metric("Lunge", f"{counters['lunge']}")

            source = pd.DataFrame({
                'Exercise': ['Pushup', 'Squat', 'Lunge'],
                'Reps': [counters['pushup'], counters['squat'], counters['lunge']]
            })

            chart = alt.Chart(source).mark_bar(color='#88c2f8').encode(
                x=alt.X('Exercise', sort=['Pushup', 'Squat', 'Lunge'], title=None, axis=alt.Axis(labelAngle=0)), 
                y=alt.Y('Reps', title=None, axis=alt.Axis(tickMinStep=1)), 
                tooltip=['Exercise', 'Reps']
            ).properties(
                height=350
            ).configure_view(
                strokeWidth=0 
            )

            st.altair_chart(chart, use_container_width=True, theme="streamlit")
            
            st.markdown("<h2 style='text-align:center;'>ประสิทธิภาพระบบ (Performance Overview)</h2>", unsafe_allow_html=True)

            total_duration = time.time() - overall_start_time
            col_time, col_fps = st.columns(2)
            col_time.metric("เวลาที่ใช้ทั้งหมด (วินาที)", f"{total_duration:.2f}")
            col_fps.metric("ความเร็วเฉลี่ย (FPS)", f"{frame_counter / total_duration:.1f}")

            st.markdown("<br>", unsafe_allow_html=True)
            st.video(final_out_mp4)

if __name__ == "__main__":
    main()
