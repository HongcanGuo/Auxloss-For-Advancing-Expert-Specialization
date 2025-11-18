# Auxloss-For-Advancing-Expert-Specialization
The code of Advancing Expert Specialization for Better MoE (NeurIPS2025 oral)

### The detailed readme.md file is coming soon.

### A temporary and brief code usage instruction

#### About the model code

Please download the base models for the following repositories from model hubs such as Hugging Face:

* [deepseek-ai/deepseek-moe-16b-chat](https://huggingface.co/deepseek-ai/deepseek-moe-16b-chat)
* [deepseek-ai/DeepSeek-V2-Lite](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite)
* [moonshotai/Moonlight-16B-A3B](https://huggingface.co/moonshotai/Moonlight-16B-A3B)

After downloading, locate the `config.json` and `modeling_deepseek.py` files within the corresponding folder under the `Model/*` path, and use them to replace the matching files in the directory of the downloaded model.

#### About the train code

Please configure according to the code in `Scripts/Train`. If computing resources are limited, we recommend using lora for fine-tuning.

#### About the inference code

Please configure according to the code in `Scripts/Inference`. If you have used LoRA for fine-tuning, please first merge LoRA with the model itself using merge.sh, and then run `Scripts/Inference/batch_vllm_infer.py`.
If LoRA is not used, simply configure the config properly and run `Scripts/Inference/batch_vllm_infer.py`.
