"""
Text-to-Video Model Evaluation Benchmark.

Stage 1: Video-Prompt consistency:
    - Generate a caption for a video
    - Calculate the similarity between the caption and the video prompt

Stage 2: Frame-Prompt consistency:
    - Generate a list of captions for each video frame using the seedbench pipeline
    - Input the captions accompanied by the initial prompt to an LLM and ask whether it matches the prompt
"""

from pathlib import Path
from argparse import ArgumentParser
from utils import load_video_frames


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--video", type=Path, default=Path("video.mp4"))
    parser.add_argument("--prompt_match_threshold", type=float, default=0.8)
    return parser.parse_args()


def main():
    config = parse_args()
    frames = load_video_frames(config.video)
    # model = SentenceTransformer(config.model)
    
    # sentence_1 = "Panda playing a guitar on times square"
    # sentence_2 = "A panda is having fun playing music with a guitar in a busy place"
    # # sentence_2 = "A guitar is being played by a panda on times square"
    # # sentence_2 = "Dog playing a guitar on times square"
    # # sentence_2 = "A panda with a guitar in a city"
    
    # emb1 = model.encode(sentence_1)
    # emb2 = model.encode(sentence_2)
    
    # similarity = util.cos_sim(emb1, emb2)
    
    # print("Cosine-Similarity:", similarity)


if __name__ == "__main__":
    main()
