import os
import torch

from transformers import BertTokenizer
from .GenerativeImage2Text import load_from_yaml_file, get_git_model, get_image_transform


class VideoCaptioner:
    def __init__(self, model_name: str, max_text_len: int=40, device="cuda", **kwargs):
        """Initialize the VideoCaptioner.

        Args:
            model_name (str): name of the model to use
            max_text_len (int, optional): maximum length of the generated text. Defaults to 40.
        """
        self.device = device
        self.params = self._load_params(model_name)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.transforms = get_image_transform(self.params)
        self.model = self._load_model(model_name)
        self.max_text_len = max_text_len

    def _load_model(self, model_name: str) -> torch.nn.Module:
        dirname = os.path.dirname(__file__)
        checkpoint_path = os.path.join(dirname, f'GenerativeImage2Text/output/{model_name}/snapshot/model.pt')
        return get_git_model(self.tokenizer, self.params, checkpoint_path, device=self.device)

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
        frames = [frame.unsqueeze(0).to(self.device) for frame in frames]

        with torch.no_grad():
            result = self.model({
                'image': frames,
            })
        caption = self.tokenizer.decode(result['predictions'][0].tolist(), skip_special_tokens=True)
        return caption
