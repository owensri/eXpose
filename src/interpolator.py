import numpy as np

class Interpolator:
    def process(self, frames_data, original_fps, target_fps):
        if not frames_data:
            return []
        
        frames_data = np.array(frames_data)
        num_frames = len(frames_data)
        duration = num_frames / original_fps
        target_frames = max(1, int(duration * target_fps))
        
        orig_indices = np.linspace(0, num_frames - 1, num=num_frames)
        target_indices = np.linspace(0, num_frames - 1, num=target_frames)
        
        resampled = []
        for col in range(frames_data.shape[1]):
            # Linear interpolation for 1D arrays
            interpolated_col = np.interp(target_indices, orig_indices, frames_data[:, col])
            resampled.append(interpolated_col)
            
        return np.column_stack(resampled).tolist()