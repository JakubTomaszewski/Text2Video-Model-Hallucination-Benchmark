import os
import re
import shutil

folders = ["models/7b/", "models/zeroscope/", "models/sd_xl/", "models/sd_v1_5/"]

for folder_path in folders:
    folder = os.listdir(folder_path)
    folder.sort()
    short_vid_folder = folder_path + "2.0/"
    long_vid_folder = folder_path + "5.0/"
    os.makedirs(short_vid_folder, exist_ok=True)
    os.makedirs(long_vid_folder, exist_ok=True)
    for video in folder:
        if video.endswith('.mp4'):
            video_name, _ = os.path.splitext(video)
            try:
                len_code = re.split(r"_", video_name)[-1]
            except IndexError:
                continue
            if len_code == "2s":
                origin = folder_path + video
                destin = short_vid_folder + video
                shutil.move(origin, destin)
            elif len_code == "5s" or len_code == "chuncks":
                origin = folder_path + video
                destin = long_vid_folder + video
                shutil.move(origin, destin)
            else:
                origin = folder_path + video
                destin = short_vid_folder + video
                shutil.move(origin, destin)
