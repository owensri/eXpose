import numpy as np

class Preprocessor:
    def __init__(self):
        # 12 Target landmarks: shoulders, elbows, wrists, hips, knees, ankles
        # เรายังคง 12 จุดนี้ไว้เหมือนเดิม เพื่อไม่ให้กระทบกับ Input Shape ของ AI Model
        self.target_indices = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

    def reset(self):
        pass

    def get_landmarks(self, results):
        if not results.pose_landmarks:
            return None
        lms = []
        # MediaPipe คืนค่า landmark มาให้ครบทั้ง 33 จุด (0-32)
        for lm in results.pose_landmarks.landmark:
            lms.append((lm.x, lm.y, lm.z, lm.visibility))
        return lms

    def get_raw_values(self, landmarks):
        raw_data = []
        for idx in self.target_indices:
            raw_data.extend(landmarks[idx])
        return raw_data

    def normalize(self, landmarks):
        # จุดสูงสุด: จมูก (0) ดึงมาให้ครบทั้งแกน X, Y และ Z เพื่อตั้งเป็นจุดศูนย์กลาง
        nose_x = landmarks[0][0]
        nose_y = landmarks[0][1]
        nose_z = landmarks[0][2]
        
        # จุดต่ำสุด (อัปเดตใหม่): นิ้วเท้าซ้าย (31) และ นิ้วเท้าขวา (32)
        left_toe_y = landmarks[31][1]
        right_toe_y = landmarks[32][1]
        
        # คำนวณความสูงโดยเฉลี่ยจากนิ้วเท้าถึงจมูก
        avg_toe_y = (left_toe_y + right_toe_y) / 2.0
        body_height = avg_toe_y - nose_y
        
        # ป้องกันการหารด้วยศูนย์ (กรณีส่วนสูงเข้าใกล้ 0 มากๆ)
        if body_height <= 0.01:
            body_height = 0.01

        norm_data = []
        # Normalize พิกัดทั้ง 12 จุดเป้าหมาย โดยอ้างอิงจุดจมูกเป็น (0,0,0)
        for idx in self.target_indices:
            x, y, z, v = landmarks[idx]
            
            # ลบจุดอ้างอิง (จมูก) ออกก่อน เพื่อแก้ปัญหาคนขยับซ้าย/ขวา/หน้า/หลัง
            nx = (x - nose_x) / body_height
            ny = (y - nose_y) / body_height
            nz = (z - nose_z) / body_height
            
            norm_data.extend([nx, ny, nz, v])
            
        return norm_data