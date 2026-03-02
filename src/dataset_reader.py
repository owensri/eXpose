import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import config

class DatasetReader:
    def __init__(self, data_path="../output_csv_norm", step_size=10, features=48):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.data_path = os.path.join(base_dir, data_path.replace('../', ''))
        
        self.max_frames = config.SEQUENCE_LENGTH
        self.classes = config.CLASSES
        self.step_size = step_size
        self.features = features
        
    def read_dataset(self):
        X_videos = []
        y_labels = []
        
        for label_idx, class_name in enumerate(self.classes):
            class_path = os.path.join(self.data_path, class_name)
            if not os.path.exists(class_path):
                print(f"[WARN] Data directory not found: {class_path}")
                continue
                
            csv_files = sorted([f for f in os.listdir(class_path) if f.endswith('.csv')])
            
            for file_name in csv_files:
                file_path = os.path.join(class_path, file_name)
                df = pd.read_csv(file_path)
                
                # ตัดคอลัมน์ที่ไม่ใช่ features ออก
                columns_to_drop = []
                if 'frame_number' in df.columns:
                    columns_to_drop.append('frame_number')
                if 'class' in df.columns:
                    columns_to_drop.append('class')
                    
                features_data = df.drop(columns=columns_to_drop, errors='ignore').values
                
                X_videos.append(features_data)
                y_labels.append(label_idx)
                
        return X_videos, y_labels

    def create_sliding_windows(self, video_data):
        windows = []
        total_frames = len(video_data)
        
        if total_frames <= self.max_frames:
            padded_window = np.zeros((self.max_frames, self.features))
            padded_window[:total_frames] = video_data
            
            if total_frames > 0 and total_frames < self.max_frames:
                padded_window[total_frames:] = video_data[-1]
                
            windows.append(padded_window)
        else:
            for start_idx in range(0, total_frames - self.max_frames + 1, self.step_size):
                end_idx = start_idx + self.max_frames
                window = video_data[start_idx:end_idx]
                windows.append(window)
                
            if (total_frames - self.max_frames) % self.step_size != 0:
                windows.append(video_data[-self.max_frames:])
                
        return windows

    def process_split(self, X_video_split, y_video_split):
        X_final = []
        y_final = []
        for i in range(len(X_video_split)):
            windows = self.create_sliding_windows(X_video_split[i])
            label = y_video_split[i]
            for w in windows:
                X_final.append(w)
                y_final.append(label)
        return np.array(X_final), np.array(y_final)

    def balance_classes(self, X, y):
        """
        [UPDATED] ฟังก์ชันปรับสมดุลข้อมูล (Undersampling ทุกคลาส)
        ทำให้ทุกคลาสมีจำนวนหน้าต่างเท่ากัน โดยใช้จำนวนขั้นต่ำเป็นโควต้า
        เพื่อป้องกัน Bias อย่างสมบูรณ์
        """
        # ถ้าไม่มีข้อมูล หรือมีคลาสเดียว ไม่ต้องทำอะไร
        if len(X) == 0 or len(self.classes) <= 1:
            return X, y

        # 1. นับจำนวนหน้าต่างของแต่ละคลาส
        class_counts = []
        for i in range(len(self.classes)):
            class_counts.append(np.sum(y == i))

        # 2. หาโควต้าเป็น "ค่าน้อยที่สุด" ของทุกคลาส
        target_quota = int(np.min(class_counts))
        print(f"  -> [Balancing] ทำให้ทุกคลาสเหลือเท่ากันที่ {target_quota} หน้าต่าง")

        X_balanced = []
        y_balanced = []

        # 3. สุ่มเลือกข้อมูลของแต่ละคลาสให้เหลือเท่ากับโควต้า
        np.random.seed(42)

        for i in range(len(self.classes)):
            idx = np.where(y == i)[0]

            if len(idx) > target_quota:
                sampled_idx = np.random.choice(idx, target_quota, replace=False)
            else:
                sampled_idx = idx

            X_balanced.extend(X[sampled_idx])
            y_balanced.extend(y[sampled_idx])

        # 4. แปลงกลับเป็น numpy array และ shuffle
        X_balanced = np.array(X_balanced)
        y_balanced = np.array(y_balanced)

        shuffle_idx = np.random.permutation(len(X_balanced))

        return X_balanced[shuffle_idx], y_balanced[shuffle_idx]

    def load_data_split(self, random_state=42):
        X_vids, y_vids = self.read_dataset()
        
        X_train_vids, X_temp_vids, y_train_vids, y_temp_vids = train_test_split(
            X_vids, y_vids, test_size=0.30, random_state=random_state, stratify=y_vids
        )
        
        X_val_vids, X_test_vids, y_val_vids, y_test_vids = train_test_split(
            X_temp_vids, y_temp_vids, test_size=0.50, random_state=random_state, stratify=y_temp_vids
        )
        
        X_train, y_train = self.process_split(X_train_vids, y_train_vids)
        X_val, y_val = self.process_split(X_val_vids, y_val_vids)
        X_test, y_test = self.process_split(X_test_vids, y_test_vids)
        
        print(f"\n[INFO] Data Splitting Complete (Windows)")
        print(f"  Train shape (Before Balance): {X_train.shape}")
        
        # ปรับสมดุลเฉพาะ TRAIN
        X_train, y_train = self.balance_classes(X_train, y_train)
        
        print(f"  Train shape (After Balance) : {X_train.shape} <--- สมดุลแล้ว!")
        print(f"  Val shape  : {X_val.shape}")
        print(f"  Test shape : {X_test.shape}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test