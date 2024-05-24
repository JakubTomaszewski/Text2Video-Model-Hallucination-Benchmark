import torch
from object_count import ObjectCounter
import json


class FrameConsistencyEvaluator:
    def __init__(self, object_counter_config: dict, prompts_path: str,
                 device: str = "cuda") -> None:

        self.object_counter = ObjectCounter(object_counter_config.image_processor_name,
                                            object_counter_config.image_model_name,
                                            object_counter_config.device)

        with open(prompts_path) as json_file:
            self.prompts_data = json.load(json_file)

    def get_objects_prompt(self, prompt: str):
        """

        Args:
            prompt (str): prompt to generate the video
        Returns:
            dict: number of objects in the video prompt
        """

        for prompt_data in self.prompts_data['prompts']:
            if prompt_data["sentence"] == prompt:
                return prompt_data["object"]

    def evaluate(self, prompt: str, frames: torch.Tensor, debug: bool = False) -> tuple[float, float]:
        """Evaluate the consistency between video frames and a prompt.

        Args:
            prompt (str): text prompt
            frames (torch.Tensor): tensor containing the frames of the video
            debug (bool): prints the object counts to debug and check the evaluation score

        Returns:
            tuple[float, float]: similarity between the generated video caption and the text prompt.
            frame_object_score corresponds to the similarity between the object instances in the video
            and the prompt.
            frame_count_score: corresponds to the similarity between the number of object instances in the video
            and the prompt.
        """

        # it supposes that the number of frames is in the first dimension
        n_frames = frames.shape[0]

        # counts the object in each frame and stores it in a dictionary
        frames_dict = {}
        for i_frame in range(n_frames):
            object_count = self.object_counter.count_objects(image=frames[i_frame])
            frames_dict[f"frame_{i_frame + 1}"] = object_count

        if debug:
            print("Object count:", frames_dict)

        # gets the objects in the prompt
        prompted_objects = self.get_objects_prompt(prompt=prompt)

        try:
            n_objects = len(prompted_objects)
        except TypeError:
            return 0.0, 0.0

        frame_object_score = 0
        frame_count_score = 0
        for frame in frames_dict:
            object_score = 0
            count_score = 0

            for prompt_object in prompted_objects.keys():
                for frame_object in frames_dict[frame]:
                    if prompt_object == frame_object:
                        object_score += 1
                        if prompted_objects[prompt_object] == frames_dict[frame][frame_object]:
                            count_score += 1
                        else:
                            continue
                    else:
                        continue
            object_score /= n_objects
            count_score /= n_objects
            frame_object_score += object_score
            frame_count_score += count_score

        frame_object_score /= n_frames
        frame_count_score /= n_frames
        return frame_object_score, frame_count_score
