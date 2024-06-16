VIDEO_CAPTIONING_MODEL_NAME="${1:-"GIT_BASE_VATEX"}"


wget https://publicgit.blob.core.windows.net/data/output/$VIDEO_CAPTIONING_MODEL_NAME/snapshot/model.pt -P src/t2vbench/evaluators/video_caption_consistency/video_captioning/GenerativeImage2Text/output/GIT_BASE_VATEX/snapshot/