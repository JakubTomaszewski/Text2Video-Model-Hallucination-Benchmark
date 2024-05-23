"""
Text-to-Video Model Evaluation Benchmark.

Stage 1: Video-Prompt consistency:
    - Generate a caption for a video
    - Calculate the similarity between the caption and the video prompt

Stage 2: Frame-Prompt consistency:
    - Generate a list of captions for each video frame using the seedbench pipeline
    - Input the captions accompanied by the initial prompt to an LLM and ask whether it matches the prompt
"""
from dotenv import load_dotenv

load_dotenv()

from config import create_parser, parse_args
from utils import load_video_frames
from text2video_consistency_evaluator import Text2VideoConsistencyEvaluator
from video_consistency.video_consistency_evaluator import VideoConsistencyEvaluator



def main():
    parser = create_parser()
    config = parse_args(parser)

    frames = load_video_frames(config.video)
    prompt = config.prompt

    video_consistency_evaluator = VideoConsistencyEvaluator(config.video_captioning,
                                                            config.sentence_similarity,
                                                            config.device)

    text2video_consistency_evaluator = Text2VideoConsistencyEvaluator(config, video_consistency_evaluator, None)
    score = text2video_consistency_evaluator.evaluate(prompt, frames, debug=config.debug)
    print(f"Consistency score: {score}")


if __name__ == "__main__":
    main()
