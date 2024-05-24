# Text-to-Video Model Hallucination Benchmark

### Data File System

```
└── data # the root folder that stores all generated videos
    └── model_name # the model that was used to run the data
        ├── 2.0 # generated videos with 2 sec duration (50 videos in total)
            ├── A baker with a white apron carefully decorates a chocolate cake with red frosting.mp4 # a video with its prompt as a name
            ├── A young boy in a bright green T-shirt is playing with a bright orange basketball on a driveway.mp4 
            └── ... 
        └── 5.0 # generated videos with 5 sec duration (50 videos in total)
            ├── A baker with a white apron carefully decorates a chocolate cake with red frosting.mp4 # a video with its prompt as a name
            ├── A young boy in a bright green T-shirt is playing with a bright orange basketball on a driveway.mp4 
            └── ... 
```