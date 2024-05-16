import os
import torch

from transformers import BertTokenizer

from .GenerativeImage2Text import load_from_yaml_file, get_git_model, get_image_transform


class VideoCaptioner:
    def __init__(self, model_name: str, max_text_len: int=40, **kwargs):
        """Initialize the VideoCaptioner.

        Args:
            model_name (str): name of the model to use
            max_text_len (int, optional): maximum length of the generated text. Defaults to 40.
        """
        self.params = self._load_params(model_name)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.transforms = get_image_transform(self.params)
        self.model = get_git_model(self.tokenizer, self.params)
        self.max_text_len = max_text_len

    def _load_params(self, model_name: str) -> dict:
        """Load the parameters for the specified model.

        Args:
            model_name (str): name of the model

        Returns:
            dict: model parameters
        """
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, f'GenerativeImage2Text/aux_data/models/{model_name}/parameter.yaml')
        return load_from_yaml_file(filename)

    def generate_caption(self, frames) -> str:
        """Generate a caption for a video.

        Args:
            video (torch.Tensor): video tensor

        Returns:
            str: generated caption
        """
        frames = [self.transforms(frame) for frame in frames]
        # frames = torch.stack(frames)
        # frames = self.transforms(frames)  # TODO: check if this also works for multiple frames
        

        with torch.no_grad():
            result = self.model({
                'image': frames,
            })
        caption = self.tokenizer.decode(result['predictions'][0].tolist(), skip_special_tokens=True)
        return caption
