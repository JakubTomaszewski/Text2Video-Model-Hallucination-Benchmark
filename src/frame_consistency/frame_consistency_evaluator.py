import torch
from .object_count import ObjectCounter


class FrameConsistencyEvaluator:
    def __init__(self, object_counter_config: dict,
                 device: str = "cuda") -> None:
        self.object_counter = ObjectCounter(**object_counter_config, device=device)

    def evaluate(self, prompt: str, frames: torch.Tensor, debug: bool = False) -> float:
        """Evaluate the consistency between video frames and a prompt.

        Args:
            prompt (str): text prompt
            frames (torch.Tensor): tensor containing the frames of the video

        Returns:
            float: similarity between the generated video caption and the text prompt
        """
        object_count = self.object_counter.count_objects(frames)
        if debug:
            print("Object count:", object_count)
        return object_count


# class VideoConsistencyEvaluator:
#     def __init__(self, video_captoning_config: dict,
#                  sentence_similarity_config: dict,
#                  device: str = "cuda") -> None:
#         self.video_captioner = VideoCaptioner(**video_captoning_config, device=device)
#         self.prompt_similarity_evaluator = SentenceSimilarityCalculator(**sentence_similarity_config, device=device)

#     def evaluate(self, prompt: str, frames: torch.Tensor, debug: bool = False) -> float:
#         """Evaluate the consistency between a video and a prompt.

#         Args:
#             prompt (str): text prompt
#             frames (torch.Tensor): tensor containing the frames of the video

#         Returns:
#             float: similarity between the generated video caption and the text prompt
#         """
#         caption = self.video_captioner.generate_caption(frames)
#         if debug:
#             print("Generated caption:", caption)
#         return self.prompt_similarity_evaluator.evaluate(caption, prompt)
