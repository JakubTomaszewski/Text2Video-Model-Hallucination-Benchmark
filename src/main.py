"""
Text-to-Video Model Evaluation Benchmark.

Stage 1: Video-Prompt consistency:
    - Generate a caption for a video
    - Calculate the similarity between the caption and the video prompt

Stage 2: Frame-Prompt consistency:
    - Generate a list of captions for each video frame using the seedbench pipeline
    - Input the captions accompanied by the initial prompt to an LLM and ask whether it matches the prompt
"""

from config import parse_args
from utils import load_video_frames
from text2video_consistency_evaluator import Text2VideoConsistencyEvaluator


def main():
    config = parse_args()

    frames = load_video_frames(config.video)
    prompt = config.prompt

    text2video_consistency_evaluator = Text2VideoConsistencyEvaluator(config)
    score = text2video_consistency_evaluator.evaluate(prompt, frames, debug=config.debug)
    print(f"Consistency score: {score}")


if __name__ == "__main__":
    main()
