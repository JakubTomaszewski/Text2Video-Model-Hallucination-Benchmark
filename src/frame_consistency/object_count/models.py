import torch


class ObjectCounter:
    def __init__(self, device: str="cuda", **kwargs) -> None:
        pass

    def count_objects(self, image: torch.Tensor) -> int:
        """Count the number of objects in a single image

        Args:
            image (torch.Tensor): image tensor

        Returns:
            int: number of objects in the image
        """
        return ...
