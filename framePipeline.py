import cv2
import os
import json
from transformers import AutoTokenizer, AutoModel
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

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


def load_image(image_file, input_size=448, max_num=6):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


path = "OpenGVLab/InternVL-Chat-V1-5-Int8"
# If you have an 80G A100 GPU, you can put the entire model on a single GPU.
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    load_in_8bit=True).eval()


def evaluate_video(video_path):
    generation_config = dict(
        num_beams=1,
        max_new_tokens=512,
        do_sample=False,
    )
    question = ("Summarize the objects in the image. Specifically, count how many times an objects appears and give the"
                "colour of the objects.")

    # Open the video file
    video = cv2.VideoCapture(video_path)
    filename, _ = os.path.splitext(video_path)
    # Specify the name of the folder you want to create
    # Create the folder
    os.makedirs(filename, exist_ok=True)
    # Check if video opened successfully
    if not video.isOpened():
        print("Error: Could not open video.")
        return

    # Get some video properties
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video.get(cv2.CAP_PROP_FPS))

    frames_dict = {}

    # Loop through each frame and save it
    for i_frame in range(frame_count):
        # Read the next frame
        ret, frame = video.read()
        if not ret:
            print(f"Error: Could not read frame {i_frame}")
            break

        frame_filename = f"{filename}/frame_{i_frame:04d}.jpg"
        cv2.imwrite(frame_filename, frame)

        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        # set the max number of tiles in `max_num`
        pixel_values = load_image(frame_filename, max_num=6).to(torch.bfloat16).cuda()

        response = model.chat(tokenizer, pixel_values, question, generation_config)

        frames_dict[f'frame_{i_frame}'] = response
    # Release the video capture object
    video.release()

    with open(f"{filename}.json", "w") as outfile:
        json.dump(frames_dict, outfile)


video_path = 'A jogger in an orange shirt runs alongside a sparkling blue lake at sunrise.mp4'
output_folder = "frames"
print("starting now...")
evaluate_video(video_path)
