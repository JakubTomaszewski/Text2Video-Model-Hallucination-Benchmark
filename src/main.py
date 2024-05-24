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
from utils import load_video_frames, rename_json_objects
from text2video_consistency_evaluator import Text2VideoConsistencyEvaluator
from video_consistency.video_consistency_evaluator import VideoConsistencyEvaluator
from frame_consistency.frame_consistency_evaluator import FrameConsistencyEvaluator


def main():
    config = parse_args()

    frames = load_video_frames(config.video)
    prompt = config.prompt

    video_consistency_evaluator = VideoConsistencyEvaluator(config.video_captioning,
                                                            config.sentence_similarity,
                                                            config.device)

    text2video_consistency_evaluator = Text2VideoConsistencyEvaluator(config, video_consistency_evaluator, None)
    score = text2video_consistency_evaluator.evaluate(prompt, frames, debug=config.debug)
    print(f"Consistency score: {score}")

    # Object consistency evaluation
    rename_json_objects(json_file_path=config.json_file_path, output_file_path=config.output_file_path)

    frame_consistency_evaluator = FrameConsistencyEvaluator(object_counter_config=config,
                                                            prompts_path=config.output_file_path,
                                                            device=config.device)

    frame_object_score, frame_count_score = frame_consistency_evaluator.evaluate(prompt, frames, config.debug)

    print(f"Unique object score: {frame_object_score}")
    print(f"Unique object score: {frame_count_score}")


if __name__ == "__main__":
    main()
