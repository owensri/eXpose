import os

# Directory settings
DATASET_FOLDER = "dataset"
RAW_CSV_FOLDER = "output_csv_raw"
NORM_CSV_FOLDER = "output_csv_norm"
REJECTED_FOLDER = "rejected_clips"

# Dataset settings
CLASSES = ["pushup", "squat", "lunge", "other"]
#CLASSES = ["other"]

# Processing & Resampling settings
TARGET_FPS = 30
TARGET_VIDEO_FRAMES = 120  # Interpolate all videos to exactly 120 frames
SEQUENCE_LENGTH = 30       # Window size (e.g., frame 0-29)
WINDOW_STEP_SIZE = 15      # Overlap by half (e.g., frame 15-44)
MIN_FRAME_COUNT = 1

# Model settings
MODEL_PATH = "exercise_model.keras"