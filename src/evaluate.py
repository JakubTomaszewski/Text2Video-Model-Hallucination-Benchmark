"""
This script is used to evaluate the consistency between a list of videos stored in a directory and their respective prompts stored in the video file names.

Process:
- Read files from a directory and run the script on each video. The prompt is the video name without the extension.
- Load the frames of the video.
- Evaluate the consistency between the text prompt and the video frames using the pipeline of tasks.
"""

import os
from dotenv import load_dotenv

load_dotenv()

from config import create_parser, parse_args
from utils import load_video_frames
from t2vbench import Text2VideoConsistencyEvaluator
from t2vbench.evaluators import VideoCaptionConsistencyEvaluator, ObjectCounter



def main():
    parser = create_parser()
    parser.add_argument('--directory', type=str, help='Name of the directory containing the files')
    config = parse_args(parser)

    video_caption_consistency_evaluator = VideoCaptionConsistencyEvaluator(config.video_captioning,
                                                                           config.sentence_similarity,
                                                                           config.device)
    object_counter = ObjectCounter(config, config.device)

    tasks = {
        "Video Caption Consistency": video_caption_consistency_evaluator,
        "Object Counting": object_counter
    }

    text2video_consistency_evaluator = Text2VideoConsistencyEvaluator(config, tasks)

    file_names = os.listdir(config.directory)
    file_paths = [os.path.join(config.directory, file) for file in file_names]

    for i, video_path in enumerate(file_paths):
        prompt = os.path.basename(video_path).strip(".mp4")
        frames = load_video_frames(video_path)

        print("-" * 50)
        print(f"Running script for video {i+1}/{len(file_paths)}: {video_path}\nprompt: {prompt}")
        score = text2video_consistency_evaluator.evaluate(prompt, frames, debug=config.debug)
        print(f"Consistency score: {score}")


if __name__ == "__main__":
    main()
