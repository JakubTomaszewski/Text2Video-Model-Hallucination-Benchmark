# Text-to-Video Model Hallucination Benchmark


## Introduction



## Environment Setup

```
$ conda env create -f environment.yml
```

or using pip

```
$ pip install -r requirements.txt
```


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

```
$ python main.py --config-file config.yaml --video path/to/video.mp4
```

