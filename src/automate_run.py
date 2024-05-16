import os
import argparse

def run_main_script(video_name, prompt):
    os.system(f'python main.py --config-file config.yaml --video "{video_name}" --prompt "{prompt}"')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Iterate through video inputs from a folder and run main script.')
    parser.add_argument('directory', type=str, help='Name of the directory containing the files')
    args = parser.parse_args()
    
    file_names = os.listdir(args.directory)
    file_names = [os.path.join(args.directory, file) for file in file_names]
    for video_name in file_names:
        prompt = os.path.basename(video_name).strip(".mp4")
        run_main_script(video_name, prompt)

