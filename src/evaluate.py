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
    file_names = [os.path.join(config.directory, file) for file in file_names]

    for i, video_name in enumerate(file_names):
        prompt = os.path.basename(video_name).strip(".mp4")
        frames = load_video_frames(os.path.join(config.directory, video_name))

        print(f"Running script for video {i+1}/{len(file_names)}: {video_name}\nprompt: {prompt}\n")
        score = text2video_consistency_evaluator.evaluate(prompt, frames, debug=config.debug)
        print(f"Consistency score: {score}")


if __name__ == "__main__":
    main()
