import cv2

class ImageProcessor:
    def convert_to_rgb(self, image):
        # MediaPipe requires RGB format
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def flip_horizontal(self, image):
        # Data augmentation (mirror effect)
        return cv2.flip(image, 1)