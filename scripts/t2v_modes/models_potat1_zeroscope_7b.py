import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_gif
from safetensors.torch import load_file
from diffusers.utils import export_to_video
import json
import argparse
import os

def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--number_of_frames", type=int, default=20, help="video length: 8 for 2 sec 20 for 5 sec")
    parser.add_argument("--output_dir", type=str, default="output", help="output folder")
    parser.add_argument("--fps", type=int, default=4, help="video length: 8 for 2 sec 20 for 5 sec")
    parser.add_argument("--input_json", type=str, default="data/prompts.json", help="input json")
    parser.add_argument("--model_name", type=str, default="potat1", help="output folder")
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

    if args.model_name == "zeroscope_v2_576w":
        pipe = DiffusionPipeline.from_pretrained("cerspense/zeroscope_v2_576w", torch_dtype=torch.float16).to(device)
        pipe.enable_model_cpu_offload()
        # memory optimization
        pipe.unet.enable_forward_chunking(chunk_size=1, dim=1)
        pipe.enable_vae_slicing()
    elif args.model_name == "text_to_video_ms_1_7b":
        pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16")
    elif args.model_name == "potat1":
        pipe = DiffusionPipeline.from_pretrained("camenduru/potat1", torch_dtype=torch.float16)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()
        pipe.enable_vae_slicing()
    pipe = pipe.to("cuda")
    

    output_dir = os.path.join(args.output_dir, args.model_name, str(args.number_of_frames/args.fps))
    os.makedirs(output_dir, exist_ok=True)
    print(output_dir)

    for i, prompt in enumerate(prompts):
        print("{}/{}: {}".format(i+1, len(prompts), prompt))
        output = pipe(prompt, num_frames=video_length, height=512, width=512)
        video_path = export_to_video(output.frames[0], fps=args.fps, output_video_path=os.path.join(output_dir, f"{prompt}.mp4"))
        # break