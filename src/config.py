import os

# Directory settings
DATASET_FOLDER = "dataset"
RAW_CSV_FOLDER = "output_csv_raw"
NORM_CSV_FOLDER = "output_csv_norm"
REJECTED_FOLDER = "rejected_clips"

# Dataset settings (Updated for 4 classes)
CLASSES = ["pushup", "squat", "lunge", "other"]
#CLASSES = ["other"]

# Processing settings
TARGET_FPS = 30
SEQUENCE_LENGTH = 100  # Matches ML window size
MIN_FRAME_COUNT = 1

# Model settings
MODEL_PATH = "exercise_model.keras"