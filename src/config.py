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
    # parser.add_argument("--prompt_match_threshold", type=float, default=0.8)

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
