import os
import sys
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, Body
import uvicorn
from collections import deque

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import config
from preprocessor import Preprocessor
from smoother import LandmarkSmoother

app = FastAPI()

# ---------------------------------------------------------
# 1. โหลดโมเดลและใช้ @tf.function เพื่อเร่งความเร็ว X10
# ---------------------------------------------------------
tf.config.set_visible_devices([], 'GPU') # บังคับใช้ CPU ให้ Overhead ต่ำสุด
project_root = os.path.dirname(current_dir)
model_path = os.path.join(project_root, config.MODEL_PATH.replace('.tflite', '.keras'))

print("--- 🚀 กำลังรัน Backend ในโหมดประสิทธิภาพสูง ---")
model = tf.keras.models.load_model(model_path, compile=False)

# แปลงฟังก์ชัน Predict ให้อยู่ในรูป Graph เพื่อความเร็วสูงสุด
@tf.function(reduce_retracing=True)
def fast_predict(input_tensor):
    return model(input_tensor, training=False)

# Warm-up
fast_predict(tf.zeros([1, config.SEQUENCE_LENGTH, 48]))
print("✅ โมเดลพร้อมทำนายแบบรวดเร็ว!")

proc = Preprocessor()
smooth = LandmarkSmoother()
sequence = deque(maxlen=config.SEQUENCE_LENGTH)

# ตัวแปร Global Cache เพื่อลดการคำนวณซ้ำซ้อน
last_action = "WAITING..."
last_confidence = 0.0
frame_count = 0

def calculate_progress(action, landmarks):
    """คำนวณ 0-100% (คำนวณแบบเร็ว ไม่กิน CPU)"""
    try:
        if action == "SQUAT":
            hip_y = (landmarks[23][1] + landmarks[24][1]) / 2
            progress = (hip_y - 0.45) / 0.35
            return max(0.0, min(1.0, progress))
        elif action == "PUSHUP":
            shoulder_y = (landmarks[11][1] + landmarks[12][1]) / 2
            progress = (shoulder_y - 0.35) / 0.35
            return max(0.0, min(1.0, progress))
    except:
        pass
    return 0.0

@app.post("/predict")
async def predict(data: list = Body(...)):
    global sequence, last_action, last_confidence, frame_count
    
    # Preprocess เร็วๆ
    smoothed_lms = smooth.process(data)
    norm_data = proc.normalize(smoothed_lms)
    sequence.append(norm_data)
    
    frame_count += 1

    # ---------------------------------------------------------
    # 2. ลดภาระ CPU: รันโมเดลทำนายแค่ 1 ครั้ง ทุกๆ 3 เฟรม
    # ---------------------------------------------------------
    if len(sequence) == config.SEQUENCE_LENGTH and frame_count % 3 == 0:
        input_data = np.expand_dims(np.array(sequence), axis=0)
        input_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)
        
        # ใช้ Fast Predict
        preds = fast_predict(input_tensor)[0]
        idx = np.argmax(preds)
        last_confidence = float(preds[idx])
        
        if last_confidence > 0.65:
            last_action = config.CLASSES[idx].upper()
        else:
            last_action = "UNCERTAIN"

    # คำนวณ % การเคลื่อนไหวให้ทุกเฟรม
    current_action = last_action if last_action != "WAITING..." else "SQUAT"
    progress = calculate_progress(current_action, data)

    return {
        "action": last_action,
        "confidence": last_confidence,
        "progress": progress,
        "frames": len(sequence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")