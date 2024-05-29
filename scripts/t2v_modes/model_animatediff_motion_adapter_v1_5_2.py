import torch
import numpy as np
import imageio
import json
from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter
from diffusers.utils import export_to_video
import json
import argparse
import os

def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--number_of_frames", type=int, default=8, help="video length: 8 for 2 sec 20 for 5 sec")
    parser.add_argument("--fps", type=int, default=4, help="video length: 8 for 2 sec 20 for 5 sec")
    parser.add_argument("--output_dir", type=str, default="output", help="output folder")
    parser.add_argument("--input_json", type=str, default="data/prompts.json", help="input json")
    parser.add_argument("--model_name", type=str, default="Animatediff_motion_adapter", help="output folder")
    args = parser.parse_args()
    return args

def load_data(input_path):
    with open(input_path, 'r') as f:
        prompt_details = json.load(f)
        prompts = [prompt['sentence'].strip(" .\n") for prompt in prompt_details['prompts']]
    return prompts

if __name__=="__main__":
    args = arguments()
    seed = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    total_frames = args.number_of_frames #8#20
    prompts = load_data(args.input_json)
    adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16)
    pipeline = AnimateDiffPipeline.from_pretrained("emilianJR/epiCRealism", motion_adapter=adapter, torch_dtype=torch.float16)
    scheduler = DDIMScheduler.from_pretrained(
        "emilianJR/epiCRealism",
        subfolder="scheduler",
        clip_sample=False,
        timestep_spacing="linspace",
        beta_schedule="linear",
        steps_offset=1,
    )
    pipeline.scheduler = scheduler
    pipeline.enable_vae_slicing()
    pipeline.enable_model_cpu_offload()

    # Chunk size (10 frames each)
    chunk_size = 10

    # Overlap size (half of chunk size)
    overlap_size = chunk_size // 2

    output_dir = os.path.join(args.output_dir, args.model_name, str(args.number_of_frames/args.fps))
    os.makedirs(output_dir, exist_ok=True)

    for j, prompt in enumerate(prompts):
        print("{}/{}: {}".format(j+1, len(prompts), prompt))
        final_frames = []
        for start_frame in range(0, total_frames, chunk_size - overlap_size):
            # Adjust context window based on chunk position
            context_window = min(chunk_size, start_frame + chunk_size)

            output = pipeline(
                prompt=prompt,
                negative_prompt="bad quality, worse quality, low resolution",
                num_frames=chunk_size,
                guidance_scale=7.5,
                num_inference_steps=50,
                generator=torch.Generator("cpu").manual_seed(49),
                context_window=context_window,  # Adjust context window size
                height=512, width=512
            )

            # Extract relevant frames (excluding overlap for first/last chunk)
            relevant_frames = output.frames[0][overlap_size:] if start_frame > 0 else output.frames[0][:chunk_size - overlap_size]
            final_frames.extend(relevant_frames)
        
        
        # video_path = export_to_video(output.frames[0], fps=4, output_video_path=os.path.join(output_dir, f"{prompt}.mp4"))
        # result1 = np.concatenate(result1)
        # result2 = [(r * 255).astype("uint8") for r in result1]
        # print(len(result2))
        # imageio.mimsave(f"{prompt}.mp4", result2, fps=4)
        print(len(final_frames))
        print(os.path.join(output_dir, f"{prompt}.mp4"))
        video_path = export_to_video(final_frames, fps=args.fps, output_video_path=os.path.join(output_dir, f"{prompt}.mp4"))
        # break
