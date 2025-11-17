import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  
import torch
import shutil
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import copy


def compare_model_weights(model1, model2):
    for name1, param1 in model1.named_parameters():
        if name1 in model2.state_dict():
            param2 = model2.state_dict()[name1]
            if not torch.allclose(param1, param2):
                print(f"Layer '{name1}': Weights are DIFFERENT.")
                return True
        else:
            print(f"Layer '{name1}' not found in the second model.")
            return True
    return False


def merge_lora(base_model_dir, lora_model_dir, output_dir):
    print(f"\nProcessing LoRA: {lora_model_dir}")
    print("Loading base model and tokenizer...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        model_base = AutoModelForCausalLM.from_pretrained(
            base_model_dir,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model_dir, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading base model: {e}")
        return

    model_base_original = copy.deepcopy(model_base)

    try:
        peft_config = PeftConfig.from_pretrained(lora_model_dir)
    except Exception as e:
        print(f"Error loading LoRA configuration: {e}")
        return

    try:
        model_lora = PeftModel.from_pretrained(
            model_base,
            lora_model_dir,
            repo_type="local",
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Error loading LoRA model: {e}")
        return

    print("Merging LoRA weights...")
    try:
        model_merged = model_lora.merge_and_unload()
    except Exception as e:
        print(f"Error merging LoRA weights: {e}")
        return

    isdifferent = compare_model_weights(model_base_original, model_merged)
    if isdifferent:
        print("Merging is valid.")
    else:
        print("Warning: Merging may be invalid.")

    print(f"Saving merged model to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    try:
        model_merged.save_pretrained(output_dir, max_shard_size="1GB")
        tokenizer.save_pretrained(output_dir)
    except Exception as e:
        print(f"Error saving merged model: {e}")
        return

    print("Done.")


def find_lora_dirs(root_dir):
    lora_dirs = []
    for root, dirs, files in os.walk(root_dir):
        for dirname in dirs:
            if dirname.startswith("saved_model_"):
                full_path = os.path.join(root, dirname)
                lora_dirs.append(full_path)
    return lora_dirs


base_model_dir = "./Models/moonshotai/Moonlight-16B-A3B"
lora_root_dir = "./kimi"
output_root_dir = "./Merged/kimi"

if not os.path.exists(base_model_dir):
    raise ValueError(f"Base model path does not exist: {base_model_dir}")

lora_paths = find_lora_dirs(lora_root_dir)

print(f"Found {len(lora_paths)} LoRA model directories.")

for lora_path in lora_paths:
    relative_path = os.path.relpath(lora_path, lora_root_dir)
    output_path = os.path.join(output_root_dir, relative_path)
    merge_lora(base_model_dir, lora_path, output_path)
