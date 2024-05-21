from datatypes import BaseEvaluator


class Text2VideoConsistencyEvaluator:
    def __init__(self,
                 config: dict,
                 video_consistency_evaluator: BaseEvaluator,
                 frame_consistency_evaluator: BaseEvaluator
                 ) -> None:
        self.device = config.device
        self.video_consistency_evaluator = video_consistency_evaluator
        self.frame_consistency_evaluator = frame_consistency_evaluator

    def evaluate(self, text_prompt, frames, debug=False):
        """Evaluate the consistency between a text prompt and video frames.

        Args:
            text_prompt (str): text prompt
            frames (torch.Tensor): tensor containing the frames of the video

        Returns:
            float: similarity between the generated caption and the text prompt
        """
        video_consistency_score = self.video_consistency_evaluator.evaluate(text_prompt, frames, debug=debug)
        # frame_consistency_score = self.frame_consistency_evaluator.evaluate(frames, text_prompt)

        return self.calculate_score(video_consistency_score, 0)

    def calculate_score(self, video_consistency_score, frame_consistency_score):
        return (video_consistency_score + frame_consistency_score) / 2
