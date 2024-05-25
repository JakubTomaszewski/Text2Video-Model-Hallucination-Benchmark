import yaml
import argparse


def create_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--config-file',
        dest='config_file',
        type=argparse.FileType(mode='r'))
    parser.add_argument("--video", type=str)
    parser.add_argument("--prompt", type=str)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--json_file_path", type=str, default="prompts.json")
    parser.add_argument("--output_file_path", type=str, default="modified_prompts.json")
    parser.add_argument("--image_processor_name", type=str, default="facebook/detr-resnet-50")
    parser.add_argument("--image_model_name", type=str, default="facebook/detr-resnet-50")
    parser.add_argument("--quantization", type=bool, default=True, help="Whether to quantize the cogvlm model or not to use less gpu resources but with less accuracy.")

    return parser


def parse_args():
    parser = create_parser()
    args = parser.parse_args()
    if args.config_file:
        data = yaml.safe_load(args.config_file)
        delattr(args, 'config_file')
        arg_dict = args.__dict__
        for key, value in data.items():
            if isinstance(value, list):
                for v in value:
                    arg_dict[key].append(v)
            else:
                arg_dict[key] = value
    return args
