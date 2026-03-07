import numpy as np

class RepCounter:
    def __init__(self, difficulty="beginner"):
        self.difficulty = difficulty.lower()
        self.counters = {'pushup': 0, 'squat': 0, 'lunge': 0}
        self.stages = {'pushup': None, 'squat': None, 'lunge': None}
        
        # Difficulty Thresholds: (Down Angle, Up Angle, Form Check Angle)
        # นำ Standard ออก เหลือแค่ Beginner กับ Advanced
        self.thresholds = {
            'beginner': {
                'pushup': (115, 140, 140), 
                'squat': (115, 140, 40),
                'lunge': (115, 140, 130)
            },
            'advanced': {
                'pushup': (90, 160, 160),
                'squat': (90, 160, 60),
                'lunge': (90, 160, 150)
            }
        }
        
        self.min_confidence = 0.80
        self.min_visibility = 0.60

    def set_difficulty(self, new_difficulty):
        self.difficulty = new_difficulty.lower()

    def calculate_angle(self, a, b, c):
        # Calculate 3D angle between 3 points
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        return 360 - angle if angle > 180.0 else angle

    def get_angles(self, landmarks, mp_pose, action):
        # Determine best side (Left or Right) based on visibility
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        
        left_vis = left_shoulder.visibility
        right_vis = right_shoulder.visibility
        
        is_left = left_vis >= right_vis

        if action == 'pushup':
            # Primary: Shoulder-Elbow-Wrist, Secondary: Shoulder-Hip-Knee
            if is_left:
                p1 = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                p2 = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
                p3 = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
                s2 = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
                s3 = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
            else:
                p1 = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                p2 = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
                p3 = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
                s2 = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
                s3 = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        else: # squat and lunge
            # Primary: Hip-Knee-Ankle, Secondary: Shoulder-Hip-Knee
            if is_left:
                p1 = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
                p2 = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
                p3 = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
                s1 = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            else:
                p1 = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
                p2 = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
                p3 = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
                s1 = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            
            s2 = p1 # Hip
            s3 = p2 # Knee

        # Check visibility
        if min(p1.visibility, p2.visibility, p3.visibility) < self.min_visibility:
            return None, None, None, None

        primary_angle = self.calculate_angle([p1.x, p1.y], [p2.x, p2.y], [p3.x, p3.y])
        
        if action == 'pushup':
            secondary_angle = self.calculate_angle([p1.x, p1.y], [s2.x, s2.y], [s3.x, s3.y])
        else:
            secondary_angle = self.calculate_angle([s1.x, s1.y], [s2.x, s2.y], [s3.x, s3.y])

        return primary_angle, secondary_angle, p2, s2

    def process(self, current_action, confidence, landmarks, mp_pose):
        feedback = "กำลังวิเคราะห์ท่าทาง..."
        color = (255, 255, 255) # White default
        
        if confidence < self.min_confidence or current_action not in self.counters:
            return self.counters, self.stages, "รอจับภาพท่าทางที่ชัดเจน...", (200, 200, 200), None

        primary_angle, secondary_angle, p_landmark, s_landmark = self.get_angles(landmarks, mp_pose, current_action)
        
        if primary_angle is None:
            # เปลี่ยนเป็น (255, 0, 0) สำหรับ RGB (สีแดง)
            return self.counters, self.stages, "อวัยวะสำคัญหลุดกล้อง!", (255, 0, 0), None

        down_th, up_th, form_th = self.thresholds[self.difficulty][current_action]

        # ---------------------------------------------------------
        # Unified Feedback Logic (ประเมินฟอร์มและการย่อตัวพร้อมกัน)
        # ---------------------------------------------------------
        is_bad_form = secondary_angle < form_th
        is_deep_down = primary_angle < down_th
        is_fully_up = primary_angle > up_th

        if is_bad_form:
            # เปลี่ยนเป็น (255, 0, 0) สำหรับ RGB (สีแดง)
            color = (255, 0, 0) 
            
            if is_deep_down:
                if current_action == 'pushup':
                    feedback = "ย่อแขนลึกดีแล้ว! แต่กดสะโพกลงให้ตัวตรง ถึงจะนับให้นะ"
                elif current_action == 'squat':
                    feedback = "ย่อลึกดีแล้ว! แต่ยืดอกขึ้นอีกนิด อย่าก้มตัว"
                elif current_action == 'lunge':
                    feedback = "ย่อลึกดีแล้ว! แต่ตั้งลำตัวให้ตรงแนวดิ่ง"
            else:
                if current_action == 'pushup':
                    feedback = "จัดฟอร์มก่อน! ทำหลังและสะโพกให้ตรงเป็นแนวเดียวกัน"
                elif current_action == 'squat':
                    feedback = "จัดฟอร์มก่อน! ยืดอกขึ้น ระวังอย่าก้มหน้า"
                elif current_action == 'lunge':
                    feedback = "จัดฟอร์มก่อน! ตั้งลำตัวให้ตรงแนวดิ่ง"
        else:
            if is_deep_down:
                self.stages[current_action] = "down"
                feedback = "ย่อลงลึกสวยมาก! ออกแรงดันตัวขึ้นเลย!"
                color = (255, 165, 0) # สีส้ม (กำลังออกแรง)
                
            elif is_fully_up:
                if self.stages[current_action] == 'down':
                    self.stages[current_action] = "up"
                    self.counters[current_action] += 1
                    feedback = "เยี่ยมมาก! นับ +1 ฟอร์มเป๊ะสุดๆ"
                    color = (0, 255, 0) # สีเขียว (สำเร็จ)
                else:
                    feedback = "ท่าเตรียมพร้อมสวยครับ เริ่มย่อตัวลงได้เลย"
                    color = (0, 255, 0) # สีเขียว (พร้อมลุย)
            else:
                if self.stages[current_action] == 'down':
                    feedback = "ดีมาก! ดันตัวขึ้นอีกให้สุดระยะ!"
                    color = (255, 165, 0) # สีส้ม
                else:
                    if current_action == 'pushup':
                        feedback = "ฟอร์มสวย! ย่อหน้าอกลงให้ลึกกว่านี้อีกนิด"
                    elif current_action == 'squat':
                        feedback = "หลังตรงดีแล้ว! ทิ้งก้นย่อลงให้ลึกกว่านี้"
                    elif current_action == 'lunge':
                        feedback = "ตัวตรงดีแล้ว! ย่อลงให้ลึกกว่านี้อีกนิด"
                    color = (150, 255, 50) # สีเขียวตองอ่อน (กำลังทำได้ดี)
        
        angle_data = {
            'primary': {'angle': primary_angle, 'landmark': p_landmark},
            'secondary': {'angle': secondary_angle, 'landmark': s_landmark}
        }
        
        return self.counters, self.stages, feedback, color, angle_data