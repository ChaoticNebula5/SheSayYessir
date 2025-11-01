import cv2

class Camera:
    def __init__(self, width=720, height=450, camera_index = 0):
        
        self.cap = cv2.VideoCapture(camera_index)
        
        if not self.cap.isOpened():
            raise RuntimeError("Camera not Found")
        
        self.width = width
        self.height = height
        
    def get_frame(self):
        
        ret, frame = self.cap.read()
        if not ret:
            return None

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (self.width, self.height))
        return frame
    
    def release(self):
        
        self.cap.release()