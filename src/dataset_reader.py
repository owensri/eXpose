import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import config

class DatasetReader:
    def __init__(self, data_path="../output_csv_norm"):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.data_path = os.path.join(base_dir, data_path.replace('../', ''))
        
        self.classes = config.CLASSES
        self.target_frames = config.TARGET_VIDEO_FRAMES
        self.window_size = config.SEQUENCE_LENGTH
        self.step_size = config.WINDOW_STEP_SIZE
        
    def interpolate_frames(self, video_data):
        # Resize any video length to exactly 120 frames using linear interpolation
        num_frames = len(video_data)
        if num_frames == self.target_frames:
            return video_data
            
        orig_indices = np.linspace(0, num_frames - 1, num=num_frames)
        target_indices = np.linspace(0, num_frames - 1, num=self.target_frames)
        
        resampled = []
        for col in range(video_data.shape[1]):
            col_data = np.interp(target_indices, orig_indices, video_data[:, col])
            resampled.append(col_data)
            
        return np.column_stack(resampled)

    def read_dataset(self):
        # Read all CSV files and interpolate them to 120 frames
        X_videos = []
        y_labels = []
        
        for label_idx, class_name in enumerate(self.classes):
            class_path = os.path.join(self.data_path, class_name)
            if not os.path.exists(class_path):
                continue
                
            csv_files = sorted([f for f in os.listdir(class_path) if f.endswith('.csv')])
            
            for file_name in csv_files:
                file_path = os.path.join(class_path, file_name)
                df = pd.read_csv(file_path)
                
                # Clean unused columns
                cols_to_drop = [c for c in ['frame_number', 'class'] if c in df.columns]
                features_data = df.drop(columns=cols_to_drop, errors='ignore').values
                
                # Force every video to be exactly 120 frames
                normalized_video = self.interpolate_frames(features_data)
                
                X_videos.append(normalized_video)
                y_labels.append(label_idx)
                
        return np.array(X_videos), np.array(y_labels)

    def create_sliding_windows(self, video_data):
        # Slice 120-frame video into 7 windows (30 frames each, step 15)
        windows = []
        for start_idx in range(0, self.target_frames - self.window_size + 1, self.step_size):
            end_idx = start_idx + self.window_size
            windows.append(video_data[start_idx:end_idx])
        return windows

    def process_split(self, X_video_split, y_video_split):
        # Apply sliding window to a group of videos
        X_final, y_final = [], []
        for i in range(len(X_video_split)):
            windows = self.create_sliding_windows(X_video_split[i])
            label = y_video_split[i]
            for w in windows:
                X_final.append(w)
                y_final.append(label)
        return np.array(X_final), np.array(y_final)

    def load_data_split(self, random_state=42):
        print(f"\n[INFO] Loading and Interpolating Videos to {self.target_frames} frames...")
        X_vids, y_vids = self.read_dataset()
        
        # Split videos into 70/15/15 (Stratified to maintain class ratios)
        X_train_vids, X_temp_vids, y_train_vids, y_temp_vids = train_test_split(
            X_vids, y_vids, test_size=0.30, random_state=random_state, stratify=y_vids
        )
        
        X_val_vids, X_test_vids, y_val_vids, y_test_vids = train_test_split(
            X_temp_vids, y_temp_vids, test_size=0.50, random_state=random_state, stratify=y_temp_vids
        )
        
        print(f"[INFO] Slicing videos into {self.window_size}-frame windows...")
        # Slice videos into windows directly (No data dropping/balancing)
        X_train, y_train = self.process_split(X_train_vids, y_train_vids)
        X_val, y_val = self.process_split(X_val_vids, y_val_vids)
        X_test, y_test = self.process_split(X_test_vids, y_test_vids)
        
        print(f"[SUCCESS] Final Data Shapes:")
        print(f"  Train shape : {X_train.shape}")
        print(f"  Val shape   : {X_val.shape}")
        print(f"  Test shape  : {X_test.shape}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test