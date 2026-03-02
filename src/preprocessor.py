import numpy as np

class Preprocessor:
    def __init__(self):
        # 12 Target landmarks: shoulders, elbows, wrists, hips, knees, ankles
        self.target_indices = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

    def reset(self):
        pass

    def get_landmarks(self, results):
        if not results.pose_landmarks:
            return None
        lms = []
        for lm in results.pose_landmarks.landmark:
            lms.append((lm.x, lm.y, lm.z, lm.visibility))
        return lms

    def get_raw_values(self, landmarks):
        raw_data = []
        for idx in self.target_indices:
            raw_data.extend(landmarks[idx])
        return raw_data

    def normalize(self, landmarks):
        nose_y = landmarks[0][1]
        left_ankle_y = landmarks[27][1]
        right_ankle_y = landmarks[28][1]
        
        # Calculate dynamic body height based on current frame
        avg_ankle_y = (left_ankle_y + right_ankle_y) / 2.0
        body_height = avg_ankle_y - nose_y
        
        # Prevent division by zero
        if body_height <= 0.01:
            body_height = 0.01

        norm_data = []
        for idx in self.target_indices:
            x, y, z, v = landmarks[idx]
            nx = x / body_height
            ny = (y - nose_y) / body_height
            nz = z / body_height
            norm_data.extend([nx, ny, nz, v])
            
        return norm_data