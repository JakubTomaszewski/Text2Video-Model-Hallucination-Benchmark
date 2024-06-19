import cv2
import torch
from PIL import Image
from torchvision.transforms import ToTensor


def load_video_frames(video_path: str):
    cap = cv2.VideoCapture(video_path)
    frames = []

    if not cap.isOpened():
        print("Error: Could not open video.")
        return frames

    # Read frames until the video ends
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert frame from BGR (OpenCV format) to RGB (PIL format)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert the NumPy array to a PIL Image
        img = Image.fromarray(frame_rgb)
        img = img.convert("RGB")
        img = ToTensor()(img)
        frames.append(img)
    
    cap.release()
    return torch.stack(frames)
