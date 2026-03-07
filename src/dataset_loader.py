import os
import config

class DatasetLoader:
    def __init__(self):
        self.dataset_path = config.DATASET_FOLDER
        self.classes = config.CLASSES

    def get_video_files(self):
        # Recursively search for videos in each class folder
        for class_name in self.classes:
            class_dir = os.path.join(self.dataset_path, class_name)
            if not os.path.exists(class_dir):
                print(f"[WARN] Directory not found: {class_dir}")
                continue

            for root, dirs, files in os.walk(class_dir):
                for file in sorted(files):
                    if file.lower().endswith(('.mp4', '.mov', '.avi')):
                        yield os.path.join(root, file), class_name, file