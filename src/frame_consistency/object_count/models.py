import torch
from transformers import DetrImageProcessor, DetrForObjectDetection


class ObjectCounter:
    def __init__(self, processor: str, model: str, device: str="cuda", **kwargs) -> None:
        """Count the number of objects of the same instance in a single image

        Args:
            processor (str): prepares the images for the object detector
            model (str): a convolutional backbone that detects objects in an image
            device (str): the device used to run the model
        """

        self.processor = DetrImageProcessor.from_pretrained(processor, revision="no_timm")
        self.model = DetrForObjectDetection.from_pretrained(model, revision="no_timm")

    def count_objects(self, image: torch.Tensor) -> dict:
        """Count the number of objects of the same instance in a single image

        Args:
            image (torch.Tensor): image tensor

        Returns:
            dict: number of objects in the image for every class
        """

        # Processes and detects the objects in the image
        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        frame_height, frame_width, _ = image.shape

        # saves the results
        target_sizes = torch.tensor([[frame_width, frame_height]])
        results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

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
