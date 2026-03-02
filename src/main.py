import os
import cv2
import config
from pose_estimator import PoseDetector
from preprocessor import Preprocessor
from interpolator import Interpolator
from image_processor import ImageProcessor
from dataset_loader import DatasetLoader
from csv_manager import CSVManager
from smoother import LandmarkSmoother

def save_sequence(data, original_fps, dir_raw, dir_norm, class_name, fname, is_flip, interpolator):
    suffix = "_flipped" if is_flip else ""
    
    # Reject sequence if it is too short
    if len(data['norm']) < config.MIN_FRAME_COUNT:
        return False

    # Resample to target FPS
    resamp_raw = interpolator.process(data['raw'], original_fps, config.TARGET_FPS)
    resamp_norm = interpolator.process(data['norm'], original_fps, config.TARGET_FPS)
    
    # Save Raw Data
    csv_raw = CSVManager(os.path.join(dir_raw, f"{fname}{suffix}.csv"))
    for r in resamp_raw:
        csv_raw.save_row(class_name, r)
    csv_raw.close()

    # Save Normalized Data
    csv_norm = CSVManager(os.path.join(dir_norm, f"{fname}{suffix}.csv"))
    for r in resamp_norm:
        csv_norm.save_row(class_name, r)
    csv_norm.close()
    
    return True

def run():
    print(f"\n[INFO] Starting Extraction Pipeline")
    print(f"[INFO] Target Classes: {', '.join(config.CLASSES)}")
    
    # Ensure directories exist
    for folder in [config.RAW_CSV_FOLDER, config.NORM_CSV_FOLDER, config.REJECTED_FOLDER]:
        os.makedirs(folder, exist_ok=True)

    loader = DatasetLoader()
    interpolator = Interpolator()
    img_processor = ImageProcessor()
    
    proc_orig, proc_flip = Preprocessor(), Preprocessor()
    smooth_orig, smooth_flip = LandmarkSmoother(), LandmarkSmoother()

    rejected_list = []
    total_videos = 0
    success_count = 0

    for video_path, class_name, video_file in loader.get_video_files():
        total_videos += 1
        
        # Instantiate separate detectors to prevent temporal state leakage
        detector_orig = PoseDetector()
        detector_flip = PoseDetector()
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"[ERROR] Cannot open {class_name}/{video_file}")
            rejected_list.append(f"{class_name}/{video_file} - File error")
            continue
            
        original_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        
        proc_orig.reset(); proc_flip.reset()
        smooth_orig.reset(); smooth_flip.reset()
        
        data_orig = {'raw': [], 'norm': []}
        data_flip = {'raw': [], 'norm': []}
        
        last_valid_orig = None
        last_valid_flip = None
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            # --- Process Original Frame ---
            rgb = img_processor.convert_to_rgb(frame)
            res = detector_orig.find_pose(rgb)
            lms = proc_orig.get_landmarks(res)
            
            if lms:
                lms = smooth_orig.process(lms)
                raw_vals = proc_orig.get_raw_values(lms)
                norm_vals = proc_orig.normalize(lms)
                
                data_orig['raw'].append(raw_vals)
                data_orig['norm'].append(norm_vals)
                last_valid_orig = {'raw': raw_vals, 'norm': norm_vals}
            elif last_valid_orig is not None:
                # Padding missing frames with the last valid pose
                data_orig['raw'].append(last_valid_orig['raw'])
                data_orig['norm'].append(last_valid_orig['norm'])

            # --- Process Flipped Frame ---
            frame_flip = img_processor.flip_horizontal(frame)
            rgb_flip = img_processor.convert_to_rgb(frame_flip)
            res_flip = detector_flip.find_pose(rgb_flip)
            lms_flip = proc_flip.get_landmarks(res_flip)
            
            if lms_flip:
                lms_flip = smooth_flip.process(lms_flip)
                raw_vals_flip = proc_flip.get_raw_values(lms_flip)
                norm_vals_flip = proc_flip.normalize(lms_flip)
                
                data_flip['raw'].append(raw_vals_flip)
                data_flip['norm'].append(norm_vals_flip)
                last_valid_flip = {'raw': raw_vals_flip, 'norm': norm_vals_flip}
            elif last_valid_flip is not None:
                data_flip['raw'].append(last_valid_flip['raw'])
                data_flip['norm'].append(last_valid_flip['norm'])
        
        # Free memory resources
        cap.release()
        detector_orig.pose.close()
        detector_flip.pose.close()

        dir_raw = os.path.join(config.RAW_CSV_FOLDER, class_name)
        dir_norm = os.path.join(config.NORM_CSV_FOLDER, class_name)
        os.makedirs(dir_raw, exist_ok=True)
        os.makedirs(dir_norm, exist_ok=True)
        
        fname = os.path.splitext(video_file)[0]
        
        # Save results
        success_orig = save_sequence(data_orig, original_fps, dir_raw, dir_norm, class_name, fname, False, interpolator)
        success_flip = save_sequence(data_flip, original_fps, dir_raw, dir_norm, class_name, fname, True, interpolator)

        if success_orig: 
            success_count += 1
            print(f"[OK] Saved: {fname}.csv")
        else: 
            rejected_list.append(f"{class_name}/{video_file} - No pose or too short (Orig)")
            print(f"[SKIP] {fname} (Orig) - Insufficient data")
            
        if success_flip: 
            success_count += 1
            print(f"[OK] Saved: {fname}_flipped.csv")
        else: 
            rejected_list.append(f"{class_name}/{video_file} - No pose or too short (Flip)")
            print(f"[SKIP] {fname} (Flip) - Insufficient data")

    # Final summary
    if rejected_list:
        reject_file_path = os.path.join(config.REJECTED_FOLDER, "rejected_log.txt")
        with open(reject_file_path, "w", encoding="utf-8") as f:
            for item in rejected_list:
                f.write(item + "\n")
                
    print(f"\n========================================")
    print(f" EXTRACTION SUMMARY")
    print(f"========================================")
    print(f"Total Videos Processed : {total_videos}")
    print(f"Target CSV Files       : {total_videos * 2}")
    print(f"Successfully Created   : {success_count}")
    print(f"Failed/Rejected Files  : {len(rejected_list)}")
    print(f"========================================\n")

if __name__ == "__main__":
    run()