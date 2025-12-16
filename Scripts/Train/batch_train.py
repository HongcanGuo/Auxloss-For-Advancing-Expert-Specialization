from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    AutoModelForSequenceClassification,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    Trainer
)
import datetime
from datasets import Dataset
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import torch

MODEL_NAME = "Moonlight-16B-A3B"
REF = "debug"
# pretrained_model_path = './Models/deepseek-ai/deepseek-moe-16b-chat'
pretrained_model_path = './Models/moonshotai/' + MODEL_NAME

timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
import os
time_stamp_path = './timestamp'
print("timestamp path:",time_stamp_path)

logging_dir = "./Output/Tensorboard/"+ MODEL_NAME + '/' + REF + f'/{timestamp}'
with open(time_stamp_path, 'w') as f:
    f.write(timestamp + '\n')
    f.write(logging_dir + '\n')
f.close()
print("timestamp:", timestamp)
print("logging dir train.py:", logging_dir)

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(pretrained_model_path, trust_remote_code=True, device_map="auto", torch_dtype=torch.bfloat16)

df = pd.read_json('./Dataset/gsm8k_train.json')
ds = Dataset.from_pandas(df)

from peft import LoraConfig, TaskType, get_peft_model

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,
    r=8, # Lora rank
    lora_alpha=32, # Lora alaph
    lora_dropout=0.1# Dropout ratio
)

model = get_peft_model(model, config)
print(model)

def process_func(example):
    MAX_LENGTH = 512 # The longer the better. Due to computing resource limitations, we used 512. Ideally, it should be stretched to 4k or 8k for best results.
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(f"<|start_header_id|>user<|end_header_id|>\n\n{example['instruction'] + example['input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False) 
    response = tokenizer(f"{example['output']}<|eot_id|>", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]  
    if len(input_ids) > MAX_LENGTH: 
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }
    
tokenized_id = ds.map(process_func, remove_columns=ds.column_names)    

training_args = TrainingArguments(
    output_dir='./results',              # output dir
    num_train_epochs=3,                  # epoch
    per_device_train_batch_size=18,       # device batch size
    warmup_steps=500,                    # schedule
    weight_decay=0.01,                   # weight decay
    logging_dir="./Output/Tensorboard/"+ MODEL_NAME + '/' + REF + f'/{timestamp}', 
    logging_steps=1,                    # logging steps
    save_steps=1000,                     # save steps
    save_total_limit=2,                  # max checkpoint num
    report_to="tensorboard"              # use tensorboard
)
print("logging_dir:", training_args.logging_dir)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_id,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)
trainer.train()
trainer.save_model(f"./SavedModels/"+ MODEL_NAME + '/' + REF + f'/saved_model_{timestamp}')
