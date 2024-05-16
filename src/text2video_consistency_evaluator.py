from video_consistency import VideoConsistencyEvaluator


class Text2VideoConsistencyEvaluator:
    def __init__(self, config: dict) -> None:
        self.device = config.device
        self.video_consistency_evaluator = VideoConsistencyEvaluator(config.video_captioning,
                                                                     config.sentence_similarity,
                                                                     config.device)
        # self.frame_consistency_evaluator = FrameConsistencyEvaluator()

    def evaluate(self, text_prompt, frames):
        """Evaluate the consistency between a text prompt and video frames.

        Args:
            text_prompt (str): text prompt
            frames (torch.Tensor): tensor containing the frames of the video

        Returns:
            float: similarity between the generated caption and the text prompt
        """
        video_consistency_score = self.video_consistency_evaluator.evaluate_video_consistency(text_prompt, frames)
        # frame_consistency_score = self.frame_consistency_evaluator.evaluate_frame_consistency(frames, text_prompt)

        return self.calculate_score(video_consistency_score, 0)

    def calculate_score(self, video_consistency_score, frame_consistency_score):
        return (video_consistency_score + frame_consistency_score) / 2
