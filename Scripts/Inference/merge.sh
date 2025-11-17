# "./Models/moonshotai/Moonlight-16B-A3B"
export BASE_MODEL_DIR="./Models/deepseek-ai/DeepSeek-V2-Lite"
export LORA_MODEL_DIR="./LoRAs/ds_v2/DeepSeek-V2-Lite/saved_model"
export MERGED_MODEL_DIR="./Merged/ds_v2"
cd ./Scripts/Inference
python lora_merge.py