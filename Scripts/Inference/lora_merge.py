import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig


def compare_model_weights(model1, model2):
    """
    Compare the weights of two models and return True as soon as any layer's weights are different (early exit).
    Return False if all weights are the same.
    """
    for name1, param1 in model1.named_parameters():
        if name1 in model2.state_dict():
            param2 = model2.state_dict()[name1]
            # Early exit if any weights are different
            if not torch.allclose(param1, param2):
                print(f"Layer '{name1}': Weights are DIFFERENT.")
                return True
        else:
            print(f"Layer '{name1}' not found in the second model.")
            return True

    # Return False if no differences were found
    return False

            
            
# Define the paths to your base model and LoRA directories
base_model_dir = os.environ.get("BASE_MODEL_DIR", None)
lora_model_dir = os.environ.get("LORA_MODEL_DIR", None)
merged_model_dir = os.environ.get("MERGED_MODEL_DIR", None)

if base_model_dir is None or lora_model_dir is None or merged_model_dir is None:
    print("Please set BASE_MODEL_DIR, LORA_MODEL_DIR and MERGED_MODEL_DIR environment variables.")
    exit(1)

print("Loading base model and tokenizer...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16
try:
    model_base = AutoModelForCausalLM.from_pretrained(base_model_dir, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model_dir, trust_remote_code=True)
except Exception as e:
    print(f"Error loading base model: {e}")
    exit(1)

import copy
model_base_original = copy.deepcopy(model_base)

print("Loading LoRA configuration...")
try:
    peft_config = PeftConfig.from_pretrained(lora_model_dir)
except Exception as e:
    print(f"Error loading LoRA configuration: {e}")
    exit(1)

print("Loading LoRA model and applying weights...")
try:
    model_lora = PeftModel.from_pretrained(
        model_base,
        lora_model_dir,
        repo_type="local", trust_remote_code=True
    )
except Exception as e:
    print(f"Error loading LoRA model: {e}")
    exit(1)

print("Merging LoRA weights into base model...")
try:
    model_merged = model_lora.merge_and_unload()
    # Now `merged_model` contains the base model with LoRA weights merged
except Exception as e:
    print(f"Error merging LoRA weights: {e}")
    exit(1)

isdifferent = compare_model_weights(model_base_original, model_merged)
if isdifferent:
    print("Merging is valid.")
else:
    print("Merging changes no params. Merging may be invalid.")

print(f"Saving merged model to {merged_model_dir}...")
try:
    model_merged.save_pretrained(merged_model_dir, max_shard_size="1GB")
    tokenizer.save_pretrained(merged_model_dir)
except Exception as e:
    print(f"Error saving merged model: {e}")
    exit(1)

print("Model merging complete.")