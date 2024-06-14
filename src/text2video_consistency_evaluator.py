class Text2VideoConsistencyEvaluator:
    """Evaluates the consistency between the text prompt and the video frames using a pipeline of tasks."""
    def __init__(self,
                 config: dict,
                 tasks: dict[str, callable],
                 ) -> None:
        """Initializes the Text2VideoConsistencyEvaluator.

        Args:
            config (dict): configuration dictionary
            tasks (dict[str, callable]): dictionary containing the tasks in the pipeline
        """
        self.device = config.device
        self.tasks: dict[str, callable] = tasks

    def evaluate(self, text_prompt, frames, debug=False) -> dict[str, float]:
        """Evaluate the consistency between the text prompt and the video frames using the tasks in the pipeline.

        Args:
            text_prompt (str): text prompt
            frames (torch.Tensor): tensor containing the frames of the video
            debug (bool, optional): whether to print debug information. Defaults to False.

        Returns:
            dict[str, float]: dictionary containing the scores for each task
        """
        return {task_name: task(text_prompt, frames, debug=debug) for task_name, task in self.tasks.items()}
