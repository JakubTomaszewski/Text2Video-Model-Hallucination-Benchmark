import torch
from diffusers import TextToVideoZeroSDXLPipeline, DDIMScheduler
from diffusers.utils import export_to_video


model_id = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = TextToVideoZeroSDXLPipeline.from_pretrained(
    model_id, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
pipe.enable_model_cpu_offload()
pipe.enable_vae_slicing()

pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
generator = torch.Generator(device="cpu").manual_seed(0)

prompt = "A dog wags its black tail excitedly as a child throws a bright yellow frisbee in the park."
result = pipe(prompt=prompt, generator=generator).images

first_frame_slice = result[0, -3:, -3:, -1]
last_frame_slice = result[-1, -3:, -3:, 0]
result = pipe(prompt=prompt).images
video_path = export_to_video(result, fps=4, output_video_path="vid.mp4")
