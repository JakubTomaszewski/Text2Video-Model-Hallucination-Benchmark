"""
This script evaluates the consistency between the generated video and the provided prompt, using a pipeline of tasks.
"""
from dotenv import load_dotenv

load_dotenv()

from config import create_parser, parse_args
from utils import load_video_frames
from t2vbench import Text2VideoConsistencyEvaluator
from t2vbench.evaluators import VideoCaptionConsistencyEvaluator, ObjectCounter



def main():
    parser = create_parser()
    config = parse_args(parser)

    frames = load_video_frames(config.video)
    prompt = config.prompt

    video_caption_consistency_evaluator = VideoCaptionConsistencyEvaluator(config.video_captioning,
                                                                           config.sentence_similarity,
                                                                           config.device)
    object_counter = ObjectCounter(config, config.device)

    tasks = {
        "Video Caption Consistency": video_caption_consistency_evaluator,
        "Object Counting": object_counter
    }

    text2video_consistency_evaluator = Text2VideoConsistencyEvaluator(config, tasks)
    results = text2video_consistency_evaluator.evaluate(prompt, frames, debug=config.debug)
    print(results)


if __name__ == "__main__":
    main()
