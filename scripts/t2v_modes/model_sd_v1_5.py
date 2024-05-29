import torch
import imageio
from diffusers import TextToVideoZeroPipeline, TextToVideoZeroSDXLPipeline

model_id = "runwayml/stable-diffusion-v1-5"
pipe = TextToVideoZeroPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
# pipe.enable_vae_slicing()

prompt = "A panda is playing guitar on times square"
result = pipe(prompt=prompt, height=512, width=512).images
result = [(r * 255).astype("uint8") for r in result]

print(len(result))
print("save .................")
imageio.mimsave("video.mp4", result, fps=4)
print("saved")