from PIL import Image
from cogvlm import CogVLM
from internvl import InternVL
import os
import json
from utils import choose_template

class ColorIdentifier:
    def __init__(self, quantization: bool, device: str="cuda", **kwargs) -> None:
        """Identify the color of the objects in the video frames.
        Args:
        """
        self.config = kwargs
        self.model_name = kwargs.color_identifier
        if self.model_name == "cogvlm":
            self.model = CogVLM(quantization, device, kwargs)
        else:
            self.model = InternVL(quantization, device, kwargs)
    
    def inference_video(self, batch, frames, prompt, prompted_colors):
        frames_dict = {}
        output_fodler = os.path.join(self.config.output_folder, self.config.color_identifier, str(self.config.template_num))
        for object_name in prompted_colors:
            snt_output_folder = os.path.join(output_fodler, prompt)
            if os.path.exists(os.path.join(snt_output_folder, f"{object_name}.json")):
                return
            object_color = prompted_colors[object_name] 
            query = choose_template(self.config.template_num, object_name, object_color)

            for i_frame, frame in enumerate(frames):
                i = i_frame + 1
                rgb_frame = frame[:, :, ::-1]  # Reverse channel order
                pil_image = Image.fromarray(rgb_frame)
                response = self.model(pil_image, query)
                frames_dict[str(i)] = response
            
            os.makedirs(snt_output_folder, exist_ok=True)
            with open(os.path.join(snt_output_folder, f"{object_name}.json"), "w") as outfile:
                    json.dump({"frames": frames_dict, "query":query, "gt_color": object_color}, outfile)
        return frames_dict
