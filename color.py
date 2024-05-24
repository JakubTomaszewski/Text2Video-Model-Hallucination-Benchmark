import cv2
import os 
import json
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, LlamaTokenizer, BitsAndBytesConfig
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
import argparse
# import random

def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, default='data/potat1/2.0', help="Path to the input folder with the videos.")
    parser.add_argument("--output_folder", type=str, default='output', help="Path to the output folder where the results will be saved.")
    parser.add_argument("--prompts", type=str,default='data/prompts.json', help="Path to the prompts file.")
    parser.add_argument("--template_num", type=int, default=2, help="Number of the template to use.")
    parser.add_argument("--batch", type=int, default=2, help="Batch size for the model.")
    parser.add_argument("--quantization", type=bool, default=True, help="Whether to quantize the model or not")
    return parser.parse_args()

def load_prompts(prompts_path):
    with open(prompts_path, 'r') as f:
        prompts = json.load(f)
    return prompts

def recur_move_to(item, tgt, criterion_func):
    if criterion_func(item):
        device_copy = item.to(tgt)
        return device_copy
    elif isinstance(item, list):
        return [recur_move_to(v, tgt, criterion_func) for v in item]
    elif isinstance(item, tuple):
        return tuple([recur_move_to(v, tgt, criterion_func) for v in item])
    elif isinstance(item, dict):
        return {k: recur_move_to(v, tgt, criterion_func) for k, v in item.items()}
    else:
        return item
    
def collate_fn(features, tokenizer) -> dict:
    images = [feature.pop('images') for feature in features]
    tokenizer.padding_side = 'left'
    padded_features = tokenizer.pad(features)
    inputs = {**padded_features, 'images': images}
    return inputs

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

    # Loop through each frame
    images = []
    frame_num = []
    for i_frame in range(frame_count):
        i = i_frame+1
        # print(i)
        ret, frame = video.read()
        if not ret:
            print(f"Error: Could not read frame {i_frame}")
            break

        rgb_frame = frame[:, :, ::-1]  # Reverse channel order
        pil_image = Image.fromarray(rgb_frame)
        images.append(pil_image)
        frame_num.append(i)
        if i % batch == 0:
            input_list = [model.build_conversation_input_ids(
                tokenizer, images=[img], query=query, history=[],
                ) for img in images]
            input_batch = collate_fn(input_list, tokenizer)
            input_batch = recur_move_to(input_batch, 'cuda', lambda x: isinstance(x, torch.Tensor))
            input_batch = recur_move_to(input_batch, precision, lambda x: isinstance(x, torch.Tensor) and torch.is_floating_point(x))

            gen_kwargs = {"max_length": 2048, "do_sample": False}
            with torch.no_grad():
                outputs = model.generate(**input_batch, **gen_kwargs)
                outputs = outputs[:, input_batch['input_ids'].shape[1]:]
                responses = tokenizer.batch_decode(outputs)
                # print(tokenizer.batch_decode(outputs))
            # print(frame_num)
            for j, response in enumerate(responses):
                frames_dict[frame_num[j]] = response
            images = []
            frame_num = []
    video.release()

    if len(frame_num) != 0:
      input_list = [model.build_conversation_input_ids(
          tokenizer, images=images, query=query, history=[],
          ) for i in range(len(frame_num))]
      input_batch = collate_fn(input_list, tokenizer)
      input_batch = recur_move_to(input_batch, 'cuda', lambda x: isinstance(x, torch.Tensor))
      input_batch = recur_move_to(input_batch, precision, lambda x: isinstance(x, torch.Tensor) and torch.is_floating_point(x))

      gen_kwargs = {"max_length": 2048, "do_sample": False}
      with torch.no_grad():
          outputs = model.generate(**input_batch, **gen_kwargs)
          outputs = outputs[:, input_batch['input_ids'].shape[1]:]
          responses = tokenizer.batch_decode(outputs)
          # print(tokenizer.batch_decode(outputs))
      for i, response in enumerate(responses):
          frames_dict[frame_num[i]] = response
    
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
        return f"Please briefly answer: If there is {article} {object_} in the image, what is its color?"
        # return f"How many times the object /'{object_}'/ appear in the image, and what colors are they? Let's think step by step."
    elif num==4:
        # Color confusion
        return f"Are there any {color} objects in the image other than {article} {object_}. Let's think step by step."

def load_model(quantization):
    if not quantization:
        with init_empty_weights():
            model = AutoModelForCausalLM.from_pretrained(
                'THUDM/cogvlm-chat-hf',
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
        device_map = infer_auto_device_map(model, max_memory={0:'15GiB','cpu':'35GiB'}, no_split_module_classes='CogVLMDecoderLayer')
        path_to_index = '/root/.cache/huggingface/hub/models--THUDM--cogvlm-chat-hf/snapshots/e29dc3ba206d524bf8efbfc60d80fc4556ab0e3c/model.safetensors.index.json'
        model = load_checkpoint_and_dispatch(
            model,
            path_to_index,   # typical, '~/.cache/huggingface/hub/models--THUDM--cogvlm-chat-hf/snapshots/balabala'
            device_map=device_map,
        )
        model = model.eval()
    else:
        print("Model with quantization")
        bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
        )
        max_memory_mapping = {0: "15GB", 'cpu': "35GB"}
        model = AutoModelForCausalLM.from_pretrained(
            'THUDM/cogvlm-chat-hf',
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            quantization_config=bnb_config,
            max_memory=max_memory_mapping
        ).eval()
    return model

if __name__ == '__main__':
    args = argument_parser()
    prompts = load_prompts(args.prompts)
    model_name = args.input_folder.split('/')[-2]
    sec = args.input_folder.split('/')[-1]
    output_fodler = os.path.join(args.output_folder, model_name, sec, str(args.template_num))
    os.makedirs(output_fodler, exist_ok=True)

    # Load the model
    tokenizer = LlamaTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5')
    model = load_model(args.quantization)

    # For each video in the input folder
    for j, prompt in enumerate(prompts["prompts"]):
        print(f"Processing video {j+1}/{len(prompts['prompts'])}: {prompt['sentence']}...")
        video_path = os.path.join(args.input_folder, prompt['sentence'] +"mp4")
        # if prompt["sentence"] != "A cat playfully chases a pink ball of yarn across a blue living room rug.":
        #     continue
        for object_name in prompt["color"]: # for every object that is colored in the sentence
            print(f"    Processing object {object_name}...")
            object_color = prompt["color"][object_name] 
            # print(object_name, object_color)
            query = choose_template(args.template_num, object_name, object_color)
            fr_dict = evaluate_video(args.batch, video_path, query, model, tokenizer, args.quantization)
            snt_output_folder = os.path.join(output_fodler, prompt['sentence'])
            os.makedirs(snt_output_folder, exist_ok=True)
            with open(os.path.join(snt_output_folder, f"{object_name}.json"), "w") as outfile:
                json.dump({"frames": fr_dict, "query":query, "gt_color": object_color}, outfile)
            # break
        # break
    print("Done!")