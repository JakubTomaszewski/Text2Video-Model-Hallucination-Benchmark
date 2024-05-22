""" Read files from a directory and run the script on each video.
The prompt is the video name without the extension.
"""

import os
from dotenv import load_dotenv

load_dotenv()

from config import create_parser, parse_args
from utils import load_video_frames
from text2video_consistency_evaluator import Text2VideoConsistencyEvaluator
from video_consistency.video_consistency_evaluator import VideoConsistencyEvaluator



def main():
    parser = create_parser()
    parser.add_argument('--directory', type=str, help='Name of the directory containing the files')
    config = parse_args(parser)

    video_consistency_evaluator = VideoConsistencyEvaluator(config.video_captioning,
                                                            config.sentence_similarity,
                                                            config.device)

    text2video_consistency_evaluator = Text2VideoConsistencyEvaluator(config,
                                                                      video_consistency_evaluator,
                                                                      None)

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
