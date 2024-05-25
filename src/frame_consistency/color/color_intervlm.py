import cv2
import os 
import json
import torch
from PIL import Image
import argparse
from transformers import AutoTokenizer, AutoModel
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, default='data/potat1/2.0', help="Path to the input folder with the videos.")
    parser.add_argument("--output_folder", type=str, default='output_int', help="Path to the output folder where the results will be saved.")
    parser.add_argument("--prompts", type=str,default='data/prompts.json', help="Path to the prompts file.")
    parser.add_argument("--template_num", type=int, default=2, help="Number of the template to use.")
    parser.add_argument("--batch", type=int, default=2, help="Batch size for the model.")
    parser.add_argument("--quantization", type=bool, default=False, help="Whether to quantize the model or not")
    return parser.parse_args()

def load_prompts(prompts_path):
    with open(prompts_path, 'r') as f:
        prompts = json.load(f)
    return prompts

def get_indefinite_article(word):
  vowels = "aeiouAEIOU"
  first_letter = word[0]
  if first_letter in vowels:
    return "an"
  else:
    return "a"
  
def evaluate_video(batch, video_path, query, model, tokenizer, quantization):
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print("Error: Could not open video.")
        return
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_dict = {}
    if quantization:
        precision = torch.float16
    else:
        precision = torch.bfloat16
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_size=448
    max_num=6
    generation_config = dict(
        num_beams=1,
        max_new_tokens=512,
        do_sample=False,
    )

    # Loop through each frame
    for i_frame in range(frame_count):
        i = i_frame+1
        # print(i)
        ret, frame = video.read()
        if not ret:
            print(f"Error: Could not read frame {i_frame}")
            break

        rgb_frame = frame[:, :, ::-1]  # Reverse channel order
        frame = Image.fromarray(rgb_frame)
        
        # rgb_frame = frame[:, :, ::-1]  # Reverse channel order
        transform = build_transform(input_size=input_size)
        images = dynamic_preprocess(frame, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values).to(precision).to(device=device) #.cuda()

        response = model.chat(tokenizer, pixel_values, query, generation_config)
        frames_dict[str(i)] = response
    video.release()
    
    return frames_dict

def choose_template(num, object_, color):
    article = get_indefinite_article(object_)
    if num==1:
        return f"Please briefly answer: What is the color of the {object_} in the image?"
    elif num==2:
        # Select a random color 
        # color = "pink"
        # colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 'black', 'white', 'brown', 'gray', 'cyan']
        # random_color = random.choice(colors)
        return f"Please briefly answer: Find the color of the {object_} in the image."
        # return f"Is the color of the {object} {random_color}? Let's think step by step."
    elif num==3:
        # Maybe the model created multiple objects with different colors
        return f"If there is {article} {object_} in the image, what is its color? Let's think step by step."
        # return f"How many times the object /'{object_}'/ appear in the image, and what colors are they? Let's think step by step."
    elif num==4:
        # Color confusion
        return f"Are there any {color} objects in the image other than {article} {object_}. Include Yes or No in your answer and a small explanation."

def build_transform(input_size):
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_model():
    path = "OpenGVLab/InternVL-Chat-V1-5-Int8"
        # If you have an 80G A100 GPU, you can put the entire model on a single GPU.
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        load_in_8bit=True).eval()
    return model

if __name__ == '__main__':
    args = argument_parser()
    prompts = load_prompts(args.prompts)
    model_name = args.input_folder.split('/')[-2]
    sec = args.input_folder.split('/')[-1]
    output_fodler = os.path.join(args.output_folder, model_name, sec, str(args.template_num))
    os.makedirs(output_fodler, exist_ok=True)

    # Load the model
    tokenizer = AutoTokenizer.from_pretrained("OpenGVLab/InternVL-Chat-V1-5-Int8", trust_remote_code=True)
    model = load_model()

    # For each video in the input folder
    for j, prompt in enumerate(prompts["prompts"]):
        print(f"Processing video {j+1}/{len(prompts['prompts'])}: {prompt['sentence']}...")
        video_path = os.path.join(args.input_folder, prompt['sentence'] +"mp4")
            
        # if prompt["sentence"] != "A cat playfully chases a pink ball of yarn across a blue living room rug.":
        #     continue
        for object_name in prompt["color"]: # for every object that is colored in the sentence
            print(f"    Processing object {object_name}...")

            snt_output_folder = os.path.join(output_fodler, prompt['sentence'])
            if os.path.exists(os.path.join(snt_output_folder, f"{object_name}.json")):
                continue

            object_color = prompt["color"][object_name] 
            # print(object_name, object_color)
            query = choose_template(args.template_num, object_name, object_color)
            fr_dict = evaluate_video(args.batch, video_path, query, model, tokenizer, args.quantization)
            os.makedirs(snt_output_folder, exist_ok=True)
            with open(os.path.join(snt_output_folder, f"{object_name}.json"), "w") as outfile:
                json.dump({"frames": fr_dict, "query":query, "gt_color": object_color}, outfile)
            # break
        # break
    print("Done!")