import os
import json
import csv
import pandas as pd


def get_objects(prompt):
    for prompt_data in prompts_data['prompts']:
        if prompt_data["sentence"] == prompt:
            return prompt_data["object"]


with open('prompts_modified.json') as json_file:
    prompts_data = json.load(json_file)

with open("score_objects.csv", 'w', newline='') as csvfile:
    # creating a csv writer object
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(["model_name", "video", "prompt", "object_score", "count_score"])

folders = ["models/potat1/", "models/7b/", "models/zeroscope/", "models/sd_xl/", "models/sd_v1_5/",
           "models/AnimateDiff_Lightning/", "models/Animatediff_motion_adapter/"]
sub_folders = ["2.0/", "5.0/"]

for folder_path in folders:
    print(f"Starting scoring of folder {folder_path}...")
    model_name = folder_path.split(sep="/")[1]
    for sub_folder_path in sub_folders:
        video_folder_path = folder_path + sub_folder_path
        video_type = sub_folder_path.split(sep="/")[0]
        video_folder = sorted([file for file in os.listdir(video_folder_path) if file.endswith('.json')])
        print("Starting scoring of a sub-folder...")
        for json_file_path in video_folder:
            video_prompt, _ = os.path.splitext(json_file_path)
            video_prompt = video_prompt.replace(" detr", "")
            video_prompt = video_prompt.split("_")[0] + "."
            # video_prompt = video_prompt.replace(" _2s", "")
            # video_prompt = video_prompt.replace(" _5s", "")

            json_file_full_path = folder_path + sub_folder_path + json_file_path
            with open(json_file_full_path) as json_file:
                frames_dict = json.load(json_file)

            prompted_objects = get_objects(prompt=video_prompt)
            try:
                n_objects = len(prompted_objects)
            except TypeError:
                print(video_prompt, "without object detected")
                with open("score_objects.csv", 'a', newline='') as csvfile:
                    # creating a csv writer object
                    csvwriter = csv.writer(csvfile)
                    csvwriter.writerow([model_name, video_type, video_prompt, 0, 0])
                    break
            n_frames = len(frames_dict)

            frame_object_score = 0
            frame_count_score = 0
            for frame in frames_dict:
                object_score = 0
                count_score = 0

                for prompt_object in prompted_objects.keys():
                    for frame_object in frames_dict[frame]:
                        if prompt_object == frame_object:
                            object_score += 1
                            if prompted_objects[prompt_object] == frames_dict[frame][frame_object]:
                                count_score += 1
                            else:
                                continue
                        else:
                            continue
                object_score /= n_objects
                count_score /= n_objects
                frame_object_score += object_score
                frame_count_score += count_score

            frame_object_score /= n_frames
            frame_count_score /= n_frames

            with open("score_objects.csv", 'a', newline='') as csvfile:
                # creating a csv writer object
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow([model_name, video_type, video_prompt, frame_object_score, frame_count_score])

df = pd.read_csv("score_objects.csv")

scores = df.groupby(by=["model_name", "video"], as_index=False).mean()
scores.to_csv('score_output.csv', index=False)
