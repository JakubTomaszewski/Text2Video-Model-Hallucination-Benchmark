import torch
from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
from diffusers.utils import export_to_gif
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from diffusers.utils import export_to_video
import json
import argparse
import os

def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--number_of_frames", type=int, default=8, help="video length: 8 for 2 sec 20 for 5 sec")
    parser.add_argument("--output_dir", type=str, default="output", help="output folder")
    parser.add_argument("--fps", type=int, default=4, help="video length: 8 for 2 sec 20 for 5 sec")
    parser.add_argument("--input_json", type=str, default="data/prompts.json", help="input json")
    parser.add_argument("--model_name", type=str, default="AnimateDiff_Lightning", help="output folder")
    args = parser.parse_args()
    return args

def load_data(input_path):
    with open(input_path, 'r') as f:
        prompt_details = json.load(f)
        prompts = [prompt['sentence'].strip(" .\n") for prompt in prompt_details['prompts']]
    return prompts
    
if __name__=="__main__":
    args = arguments()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16
    video_length = args.number_of_frames #8#20
    prompts = load_data(args.input_json)

    step = 4  # Options: [1,2,4,8]
    repo = "ByteDance/AnimateDiff-Lightning"
    ckpt = f"animatediff_lightning_{step}step_diffusers.safetensors"
    base = "emilianJR/epiCRealism"  # Choose to your favorite base model.
    adapter = MotionAdapter().to(device, dtype)

    adapter.load_state_dict(load_file(hf_hub_download(repo ,ckpt), device=device))
    pipe = AnimateDiffPipeline.from_pretrained(base, motion_adapter=adapter, torch_dtype=dtype).to(device)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing", beta_schedule="linear")
    pipe.enable_vae_slicing()

    output_dir = os.path.join(args.output_dir, args.model_name, str(args.number_of_frames/args.fps))
    os.makedirs(output_dir, exist_ok=True)

    for i, prompt in enumerate(prompts):
        print("{}/{}: {}".format(i+1, len(prompts), prompt))
        output = pipe(prompt=prompt, guidance_scale=1.0,
        num_inference_steps=step, num_frames=video_length, height=512, width=512)
        video_path = export_to_video(output.frames[0], fps=args.fps, output_video_path=os.path.join(output_dir, f"{prompt}.mp4"))
        # break