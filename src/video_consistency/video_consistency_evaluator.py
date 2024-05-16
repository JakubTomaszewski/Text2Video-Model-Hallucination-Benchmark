from sentence_similarity import SentenceSimilarityCalculator
from video_captioning import VideoCaptioner


class VideoConsistencyEvaluator:
    def __init__(self):
        self.video_captioner = VideoCaptioner(model_name="GIT_BASE_VATEX")  # TODO: load model name from config
        self.sentence_similarity_calculator = SentenceSimilarityCalculator(model_name="all-MiniLM-L6-v2")  # TODO: load model name from config

    def evaluate_video_consistency(self, video, prompt):
        """Evaluate the consistency between a video and a prompt.

        Args:
            video (torch.Tensor): video tensor
            prompt (str): prompt

        Returns:
            float: similarity between the video caption and the prompt
        """
        caption = self.video_captioner.generate_caption(video)
        return self.sentence_similarity_calculator.calculate_similarity(caption, prompt)
