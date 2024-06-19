import json
import torch
from transformers import DetrImageProcessor, DetrForObjectDetection


class ObjectCounter:
    def __init__(self, config: dict, device: str = "cuda") -> None:
        """Initializes the object counter

        Args:
            config (dict): configuration for the object counter model
            device (str, optional): device to run the object counter model. Defaults to "cuda".
        """
        self.img_processor = DetrImageProcessor.from_pretrained(config.image_processor_name, revision="no_timm")
        self.model = DetrForObjectDetection.from_pretrained(config.image_model_name, revision="no_timm")
        self.device = device

        # Load prompts
        with open(config.prompts_file_path) as json_file:
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


    def count_objects(self, image: torch.Tensor) -> dict:
        """Count the number of objects of the same instance in a single image

        Args:
            image (torch.Tensor): image tensor

        Returns:
            dict: number of objects in the image for every class
        """

        # Processes and detects the objects in the image
        inputs = self.img_processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        frame_height, frame_width, _ = image.shape

        # saves the results
        target_sizes = torch.tensor([[frame_width, frame_height]])
        results = self.img_processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.7)[0]
        print("Predicted labels:", results["labels"])

        # dictionary that counts the appearance of every object in an image
        count_objects_image = {}
        for label in results["labels"]:
            label_name = self.model.config.id2label[label.item()]

            # if the object is not already detected, creates the key
            if label_name not in count_objects_image:
                count_objects_image[label_name] = 1

            # if the key already exists add one to the count
            else:
                count_objects_image[label_name] += 1
        return count_objects_image

    def evaluate(self, prompt: str, frames: torch.Tensor, debug: bool = False) -> tuple[float, float]:
        """Evaluate the consistency between video frames and a prompt.

        Args:
            prompt (str): text prompt
            frames (torch.Tensor): tensor containing the frames of the video
            debug (bool): prints the object counts to debug and check the evaluation score

        Returns:
            tuple[float, float]: scores for the object count and the object instances in the video. Includes:
                - frame_object_score: score corresponding to the occurrence of the object instances in the video and the prompt.
                - frame_count_score: score corresponding to the number of object instances in the video and the prompt.
        """

        # it supposes that the number of frames is in the first dimension
        n_frames = frames.shape[0]

        # counts the object in each frame and stores it in a dictionary
        frames_dict = {}
        for i, frame in enumerate(frames):
            object_count = self.count_objects(image=frame)
            frames_dict[f"frame_{i + 1}"] = object_count

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
    
    def __call__(self, prompt: str, frames: torch.Tensor, debug: bool = False) -> tuple[float, float]:
        return self.evaluate(prompt, frames, debug)
