import cv2
import os
import json
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch

processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")


def extract_frames(video_path):

    # Open the video file
    video = cv2.VideoCapture(video_path)
    filename, _ = os.path.splitext(video_path)
    filename = filename + " detr"
    # Check if video opened successfully
    if not video.isOpened():
        print("Error: Could not open video.")
        return

    # Get some video properties
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    frames_dict = {}
    # Loop through each frame and save it
    for i_frame in range(frame_count):
        # Read the next frame
        ret, frame = video.read()
        if not ret:
            print(f"Error: Could not read frame {i_frame + 1}")
            break

        # Save the frame
        frames_dict[f'frame_{i_frame + 1}'] = {}
        inputs = processor(images=frame, return_tensors="pt")
        outputs = model(**inputs)
        frame_height, frame_width, _ = frame.shape
        target_sizes = torch.tensor([[frame_width, frame_height]])
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

        for label in results["labels"]:
            label_name = model.config.id2label[label.item()]
            if label_name not in frames_dict[f'frame_{i_frame+1}']:
                frames_dict[f'frame_{i_frame + 1}'][label_name] = 1
            else:
                frames_dict[f'frame_{i_frame + 1}'][label_name] += 1

    # Release the video capture object
    video.release()

    with open(f"{filename}.json", "w") as outfile:
        json.dump(frames_dict, outfile)


video_path = 'A family of three builds a snowman with a bright orange carrot nose in their snowy front yard.mp4'

folders = ["models/potat1/", "models/7b/", "models/zeroscope/", "models/sd_xl/", "models/sd_v1_5/",
           "models/AnimateDiff_Lightning/", "models/Animatediff_motion_adapter/"]

sub_folders = ["2.0/", "5.0/"]
for folder_path in folders:
    for sub_folder_path in sub_folders:
        video_folder_path = folder_path + sub_folder_path
        video_folder = os.listdir(video_folder_path)
        video_folder.sort()

        for video in video_folder:
            print("Video name is", video)
            if video.endswith('.mp4'):
                video_path = video_folder_path + video
                extract_frames(video_path)
