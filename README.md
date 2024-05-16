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


## Model Download

The `GIT_BASE_VATEX` model checkpoints used for video captioning can be downloaded by following this link: https://publicgit.blob.core.windows.net/data/output/GIT_BASE_VATEX/snapshot/model.pt. This `model.pt` file should be placed in `Text2Video-Model-Hallucination-Benchmark/src/video_consistency/video_captioning/GenerativeImage2Text/output/GIT_BASE_VATEX/snapshot/`.
```
$ python main.py --config-file config.yaml --video path/to/video.mp4
```


## Runnning the code

```
$ python main.py --config-file config.yaml --video path/to/video.mp4
```

