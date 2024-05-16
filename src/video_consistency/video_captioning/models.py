import torch

from transformers import BertTokenizer

from .GenerativeImage2Text.generativeimage2text.tsv_io import load_from_yaml_file
from .GenerativeImage2Text.generativeimage2text.model import get_git_model
from .GenerativeImage2Text.generativeimage2text.inference import get_image_transform


class VideoCaptioner:
    def __init__(self, model_name: str, prefix: str):
        self.params = load_from_yaml_file(f'aux_data/models/{model_name}/parameter.yaml')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.transforms = get_image_transform(self.params)
        self.model = get_git_model(self.tokenizer, self.params)
        self.prefix = prefix  # TODO: check if it works without the prefix
        self.max_text_len = 40

    def prepare_prefix(self, prefix):
        prefix_encoding = self.tokenizer(prefix,
                                        padding='do_not_pad',
                                        truncation=True,
                                        add_special_tokens=False,
                                        max_length=self.max_text_len)
        payload = prefix_encoding['input_ids']
        if len(payload) > self.max_text_len - 2:
            payload = payload[-(self.max_text_len - 2):]
        input_ids = [self.tokenizer.cls_token_id] + payload
        return input_ids


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
        
        input_ids = self.prepare_prefix(self.prefix)  # TODO: check if it is required

        with torch.no_grad():
            result = self.model({
                'image': frames,
                'prefix': torch.tensor(input_ids).unsqueeze(0).cuda(),
            })
        caption = self.tokenizer.decode(result['predictions'][0].tolist(), skip_special_tokens=True)
        return caption
