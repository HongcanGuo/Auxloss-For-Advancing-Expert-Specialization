import json
from vllm import LLM, SamplingParams
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

base_model_path = "./Models/deepseek-ai/DeepSeek-V2-Lite"
max_model_len = 1024
gpu_memory_utilization = 0.9  


llm = LLM(model=base_model_path,
          gpu_memory_utilization=gpu_memory_utilization, trust_remote_code=True, tensor_parallel_size=2)

sampling_params = SamplingParams(temperature=0, max_tokens=max_model_len)

def read_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def batch_inference(data, output_file):
    instructions = [item["instruction"] for item in data]
    outputs = llm.generate(
        instructions,
        sampling_params
    )
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, (output, item) in enumerate(zip(outputs, data), start=1):
            prompt = output.prompt
            generated_text = output.outputs[0].text
            result = {
                "id": i,
                "question": prompt,
                "original_answer": item["output"],
                "response": generated_text
            }
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    file_paths = [
        "./Datasets/gsm8k.json",
        "./Datasets/math500.json",
        # You could add more benchmarks
    ]

    for file_path in file_paths:
        file_name = os.path.basename(file_path).split('.')[0]
        output_file = f"./{file_name}.jsonl"
        data = read_dataset(file_path)
        batch_inference(data, output_file)    