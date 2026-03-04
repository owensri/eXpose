import streamlit as st
import os
 # Force TensorFlow to use legacy tf.keras instead of Keras 3 (required for models trained with TF 2.x)
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import sys
import tempfile
import cv2
import numpy as np
import mediapipe as mp
import time
import subprocess
from collections import deque
from tensorflow.keras.models import load_model
from functools import lru_cache

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
    page_icon=None,
    layout="wide"
)

# ---------------------------------------------------------
# Custom CSS for Clean Premium Layout
# ---------------------------------------------------------
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    h1 {
        font-weight: 600;
        letter-spacing: -0.5px;
    }
    h2, h3 {
        font-weight: 500;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    div[data-testid="stMetric"] {
        background-color: #111111;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
    }
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 600;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    </style>
""", unsafe_allow_html=True)

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
# Production Threshold Configuration
# ---------------------------------------------------------
CONFIDENCE_THRESHOLD = 0.80
OTHER_THRESHOLD = 0.85
VISIBILITY_THRESHOLD = 0.6
VOTING_WINDOW = 5

# ---------------------------------------------------------
# Model Caching (Production Optimization)
# ---------------------------------------------------------
@st.cache_resource
def load_model_cached(path):
    return load_model(path, compile=False)

# ---------------------------------------------------------
# Dynamic Side Angle Selection
# ---------------------------------------------------------
def get_best_leg_angle(landmarks, mp_pose):
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
    left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]

    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
    right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
    right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]

    left_visible = min(left_hip.visibility, left_knee.visibility, left_ankle.visibility)
    right_visible = min(right_hip.visibility, right_knee.visibility, right_ankle.visibility)

    if left_visible < VISIBILITY_THRESHOLD and right_visible < VISIBILITY_THRESHOLD:
        return None

    if left_visible >= right_visible:
        a = [left_hip.x, left_hip.y]
        b = [left_knee.x, left_knee.y]
        c = [left_ankle.x, left_ankle.y]
    else:
        a = [right_hip.x, right_hip.y]
        b = [right_knee.x, right_knee.y]
        c = [right_ankle.x, right_ankle.y]

    return calculate_angle(a, b, c)

def get_best_arm_angle(landmarks, mp_pose):
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]

    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]

    left_visible = min(left_shoulder.visibility, left_elbow.visibility, left_wrist.visibility)
    right_visible = min(right_shoulder.visibility, right_elbow.visibility, right_wrist.visibility)

    if left_visible < VISIBILITY_THRESHOLD and right_visible < VISIBILITY_THRESHOLD:
        return None

    if left_visible >= right_visible:
        a = [left_shoulder.x, left_shoulder.y]
        b = [left_elbow.x, left_elbow.y]
        c = [left_wrist.x, left_wrist.y]
    else:
        a = [right_shoulder.x, right_shoulder.y]
        b = [right_elbow.x, right_elbow.y]
        c = [right_wrist.x, right_wrist.y]

    return calculate_angle(a, b, c)

# ---------------------------------------------------------
# 3. Main Function
# ---------------------------------------------------------
def main():
    st.markdown("<h1 style='text-align: center;'>AI Exercise Analysis System</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #888888;'>Exercise movement classification and repetition counting using artificial intelligence.</p>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # ==========================================
    # Sidebar
    # ==========================================
    with st.sidebar:
        st.header("System Configuration")
        
        st.subheader("1. Model Selection")
        model_choice = st.selectbox(
            "Select Model:",
            ["LSTM + CNN", 
             "CNN", 
             "LSTM"]
        )
        
        model_files = {
            "LSTM + CNN": "exercise_model_hybrid.keras",
            "CNN": "exercise_model_cnn.keras",
            "LSTM": "exercise_model_lstm.keras"
        }
        selected_model_file = model_files[model_choice]
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, selected_model_file)

        st.markdown("---")

        st.subheader("2. Processing Mode")
        display_mode = st.radio(
            "Select Processing Mode:",
            ["Fast Mode", 
             "Live Preview"]
        )
        is_fast_mode = "Fast Mode" in display_mode

        st.markdown("---")

        st.subheader("3. Upload Video")
        uploaded_file = st.file_uploader(
            "เลือกไฟล์วิดีโอ (MP4, MOV, AVI)", 
            type=['mp4', 'mov', 'avi']
        )

        st.markdown("---")
        start_button = st.button("Start Processing", use_container_width=True, type="primary")

    # ==========================================
    # Main Area
    # ==========================================
    if uploaded_file is None:
        st.info("Please upload a video file and configure the settings from the sidebar to begin analysis.")
    else:
        if not start_button:
            st.success(f"File '{uploaded_file.name}' uploaded successfully. Click 'Start Processing' to begin analysis.")
            st.video(uploaded_file)
        else:
            st.subheader(f"Processing Video (Model: {model_choice})")
            
            if not os.path.exists(model_path):
                st.error(f"Model file '{selected_model_file}' not found. Please ensure the model file is located in the same directory as app.py.")
                return

            overall_start_time = time.time()

            model = load_model_cached(model_path)
            classes = config.CLASSES
            
            mp_pose = mp.solutions.pose
            pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
            
            preprocessor = Preprocessor()
            smoother = LandmarkSmoother()
            window_frames = deque(maxlen=config.SEQUENCE_LENGTH)
            prediction_buffer = deque(maxlen=VOTING_WINDOW)

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

            # --- ตัวแปรสำหรับนับรอบ (Repetition Counters แยกรายท่า) ---
            counters = {'pushup': 0, 'squat': 0, 'lunge': 0}
            stages = {'pushup': None, 'squat': None, 'lunge': None}
            
            # กำหนดเกณฑ์ความมั่นใจขั้นต่ำ (80%) 
            CONFIDENCE_THRESHOLD = 0.80

            if is_fast_mode:
                video_placeholder.info("Processing video in Fast Mode...")

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

                            if "other" in classes:
                                other_idx = classes.index("other")
                                if best_class_idx == other_idx and predictions[other_idx] < OTHER_THRESHOLD:
                                    temp_preds = predictions.copy()
                                    temp_preds[other_idx] = 0.0
                                    best_class_idx = np.argmax(temp_preds)

                            prediction_buffer.append(best_class_idx)

                            if len(prediction_buffer) == prediction_buffer.maxlen:
                                voted_class_idx = max(set(prediction_buffer), key=prediction_buffer.count)
                            else:
                                voted_class_idx = best_class_idx

                            confidence = predictions[voted_class_idx]
                            current_action = classes[voted_class_idx]

                    # 2. ระบบนับจำนวนครั้ง (Hybrid Logic: นับเมื่อ AI มั่นใจ > 80% และทำครบ Down -> Up)
                    try:
                        if confidence >= CONFIDENCE_THRESHOLD:

                            if current_action == 'squat':
                                angle = get_best_leg_angle(landmarks, mp_pose)
                                if angle is not None:
                                    if angle < 115:
                                        stages['squat'] = "down"
                                    if angle > 140 and stages['squat'] == 'down':
                                        stages['squat'] = "up"
                                        counters['squat'] += 1

                            elif current_action == 'pushup':
                                angle = get_best_arm_angle(landmarks, mp_pose)
                                if angle is not None:
                                    if angle < 110:
                                        stages['pushup'] = "down"
                                    if angle > 140 and stages['pushup'] == 'down':
                                        stages['pushup'] = "up"
                                        counters['pushup'] += 1

                            elif current_action == 'lunge':
                                angle = get_best_leg_angle(landmarks, mp_pose)
                                if angle is not None:
                                    if angle < 115:
                                        stages['lunge'] = "down"
                                    if angle > 140 and stages['lunge'] == 'down':
                                        stages['lunge'] = "up"
                                        counters['lunge'] += 1

                    except Exception:
                        pass

                # --- 3. แสดงผล UI บนวิดีโอ ---
                
                # ดึงสถานะปัจจุบันของท่าที่กำลังทำอยู่มาโชว์ (ถ้าเป็น None ให้เป็น "-")
                current_stage = stages.get(current_action) or "-"
                
                # แถบบอกท่าทางด้านซ้ายบน
                text_color = (0, 255, 0) if confidence > CONFIDENCE_THRESHOLD else (255, 165, 0)
                if len(window_frames) < config.SEQUENCE_LENGTH:
                    display_text = f"Buffering... {len(window_frames)}/{config.SEQUENCE_LENGTH}"
                    text_color = (255, 255, 0)
                else:
                    display_text = f"Action: {current_action.upper()} ({confidence*100:.1f}%) | State: {current_stage.upper()}"

                cv2.rectangle(image_rgb, (0, 0), (800, 45), (0, 0, 0), -1)
                cv2.putText(image_rgb, display_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2, cv2.LINE_AA)

                # กล่องนับจำนวนครั้งแยก 3 ท่า (Reps Counters) ด้านขวาบน
                cv2.rectangle(image_rgb, (620, 0), (800, 110), (40, 40, 40), -1) # พื้นหลังสีเทาเข้ม
                
                # โชว์ PUSHUP
                cv2.putText(image_rgb, f"PUSHUP: {counters['pushup']}", (630, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0) if current_action == 'pushup' else (255,255,255), 2, cv2.LINE_AA)
                # โชว์ SQUAT
                cv2.putText(image_rgb, f"SQUAT : {counters['squat']}", (630, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0) if current_action == 'squat' else (255,255,255), 2, cv2.LINE_AA)
                # โชว์ LUNGE
                cv2.putText(image_rgb, f"LUNGE : {counters['lunge']}", (630, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0) if current_action == 'lunge' else (255,255,255), 2, cv2.LINE_AA)

                # บันทึกเฟรมที่วาดเสร็จแล้ว
                writer.write(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))

                if is_fast_mode:
                    if frame_counter % 15 == 0:
                        if total_frames > 0:
                            progress = min(frame_counter / total_frames, 1.0)
                            progress_bar.progress(progress)
                            status_text.text(f"Processing in Fast Mode... {int(progress * 100)}%")
                else:
                    if frame_counter % 5 == 0:
                        video_placeholder.image(image_rgb, channels="RGB", use_container_width=True)
                        if total_frames > 0:
                            progress = min(frame_counter / total_frames, 1.0)
                            progress_bar.progress(progress)
                            status_text.text(f"Processing in Live Preview Mode... {int(progress * 100)}%")

            cap.release()
            writer.release()
            
            # ---------------------------------------------------------
            # เข้ารหัสวิดีโอ
            # ---------------------------------------------------------
            status_text.text("Finalizing processed video output...")
            cmd = ['ffmpeg', '-y', '-i', temp_out_mp4, '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', '-preset', 'fast', final_out_mp4]
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception as e:
                st.warning("Video encoding failed. Displaying the original processed file instead.")
                final_out_mp4 = temp_out_mp4

            overall_end_time = time.time()
            total_duration = overall_end_time - overall_start_time

            video_placeholder.empty()
            progress_bar.empty()
            status_text.empty()

            # ------------------- Custom Results Section -------------------
            st.success("Processing completed successfully.")

            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown("<h2 style='text-align:center;'>Repetition Dashboard</h2>", unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Pushup", f"{counters['pushup']}")
            with col2:
                st.metric("Squat", f"{counters['squat']}")
            with col3:
                st.metric("Lunge", f"{counters['lunge']}")

            st.markdown("<br>", unsafe_allow_html=True)

            # Bar Chart Visualization
            chart_data = {
                "Pushup": counters['pushup'],
                "Squat": counters['squat'],
                "Lunge": counters['lunge']
            }
            st.bar_chart(chart_data)

            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown("<h2 style='text-align:center;'>Performance Overview</h2>", unsafe_allow_html=True)

            col_time, col_fps = st.columns(2)
            with col_time:
                st.metric("Total Processing Time (seconds)", f"{total_duration:.2f}")
            with col_fps:
                st.metric("Average Processing Speed (FPS)", f"{frame_counter / total_duration:.1f}")

            st.markdown("<br>", unsafe_allow_html=True)

            st.video(final_out_mp4)

if __name__ == "__main__":
    main()
