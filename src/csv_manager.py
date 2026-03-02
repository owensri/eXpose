import csv

class CSVManager:
    def __init__(self, filepath):
        self.filepath = filepath
        self.file = open(self.filepath, 'w', newline='')
        self.writer = csv.writer(self.file)
        self.is_header_written = False
        self.frame_count = 1

        # Map landmark ID with its descriptive name (MediaPipe Topology)
        self.landmarks = [
            (11, "left_shoulder"), (12, "right_shoulder"),
            (13, "left_elbow"),    (14, "right_elbow"),
            (15, "left_wrist"),    (16, "right_wrist"),
            (23, "left_hip"),      (24, "right_hip"),
            (25, "left_knee"),     (26, "right_knee"),
            (27, "left_ankle"),    (28, "right_ankle")
        ]

    def save_row(self, class_name, features):
        # Write header on the first row
        if not self.is_header_written:
            header = ['frame_number', 'class']
            
            # Generate descriptive column names e.g., 'left_shoulder_11_x'
            for lm_id, name in self.landmarks:
                header.extend([
                    f'{name}_{lm_id}_x', 
                    f'{name}_{lm_id}_y', 
                    f'{name}_{lm_id}_z', 
                    f'{name}_{lm_id}_v'
                ])
                
            self.writer.writerow(header)
            self.is_header_written = True
        
        # Write data row
        row = [self.frame_count, class_name] + list(features)
        self.writer.writerow(row)
        
        # Increment frame counter for the next row
        self.frame_count += 1

    def close(self):
        if self.file:
            self.file.close()