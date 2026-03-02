class LandmarkSmoother:
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.prev_landmarks = None

    def reset(self):
        # Reset state for new video sequence
        self.prev_landmarks = None

    def process(self, landmarks):
        if not landmarks:
            return None
        
        if self.prev_landmarks is None:
            self.prev_landmarks = landmarks
            return landmarks
        
        smoothed = []
        # Exponential Moving Average (EMA) for jitter reduction
        for curr, prev in zip(landmarks, self.prev_landmarks):
            x = self.alpha * curr[0] + (1 - self.alpha) * prev[0]
            y = self.alpha * curr[1] + (1 - self.alpha) * prev[1]
            z = self.alpha * curr[2] + (1 - self.alpha) * prev[2]
            v = curr[3]
            smoothed.append((x, y, z, v))
        
        self.prev_landmarks = smoothed
        return smoothed