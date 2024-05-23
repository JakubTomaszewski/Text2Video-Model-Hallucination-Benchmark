# Text-to-Video Model Hallucination Benchmark


## Table of Contents

- [Introduction](#introduction)
- [Environment Setup](#environment-setup)
  * [Requirements Installation](#requirements-installation)
  * [Environment variables](#environment-variables)
- [Model Checkpoints](#model-checkpoints)
- [Runnning the code](#runnning-the-code)
  * [Evaluation on a single video](#evaluation-on-a-single-video)
  * [Evaluation on a directory of videos](#evaluation-on-a-directory-of-videos)


## Introduction



## Environment Setup

### Requirements Installation

```
$ conda env create -f environment.yml
```

or using pip

```
$ pip install -r requirements.txt
```

Then activate the environment:

```
$ conda activate text2video-benchmark
```


### Environment variables

By default the project uses the [Meta Instruct Llama 3](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) model for assessing whether the initial video prompt matches the caption generated by the video captioning model. To use this model, you need to request access to the model from the Hugging Face model hub and set the `HF_TOKEN` environment variable to your Hugging Face API token (use [this](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) link to generate your own token). This can be achieved by running the following command:

```
$ export HF_TOKEN="your_hugging_face_token"
```

or by creating a `.env` file in the root directory of the project and adding the following line:

```
HF_TOKEN="your_hugging_face_token"
```

the `.env` file will be automatically loaded by the project.


## Model Checkpoints

A quick download of the model checkpoints can be performed by running the `download_checkpoints.sh` script located in the [scripts](scripts) directory. The script will download the model checkpoints for the different models used in the benchmark and place them in the appropriate directories.

If no `MODEL_NAME` is provided, the script will download the checkpoint for the `GIT_BASE_VATEX` model.
```
$ sh scripts/download_checkpoints.sh MODEL_NAME
```


If you wish to download the model checkpoints individually, you can do so by following the instructions below:

- Video Captioning:

The `GIT_BASE_VATEX` model checkpoint used for video captioning can be downloaded using the following [link](https://publicgit.blob.core.windows.net/data/output/GIT_BASE_VATEX/snapshot/model.pt). The downloaded checkpoint file, namely `model.pt`m should be placed in the following directory: `src/video_consistency/video_captioning/GenerativeImage2Text/output/GIT_BASE_VATEX/snapshot/`.



## Runnning the code

### Evaluation on a single video

To run the benchmark on a single video, you can execute the following command from the root directory of the project:

```
$ python src/main.py --config-file src/config.yaml --video path/to/video.mp4 --prompt "initial video prompt"
```

The `config.yaml` file contains the configuration parameters for the benchmark. The `video` argument specifies the path to the video file that will be used as input for the benchmark.


### Evaluation on a directory of videos

To run the benchmark on a directory of videos, you can execute the following command from the root directory of the project:

```
$ python src/evaluate.py --config-file src/config.yaml --directory path/to/directory/with/videos
```

**Note:** The `evaluate.py` script assumes that the video files are named based on their initial prompt used to generate the video and will read the prompt automatically from the video file name.
For example, if the initial prompt used to generate the video was `"A cat playing with a ball"`, the video file should be named `"A cat playing with a ball.mp4"`.

The `config.yaml` file contains the configuration parameters for the benchmark. The `directory` argument specifies the path to the directory containing the video files that will be used as input for the benchmark.