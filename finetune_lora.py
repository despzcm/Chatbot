#-*- coding:utf-8 -*-
import argparse
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import sys
import torch
from transformers import TrainingArguments, HfArgumentParser, Trainer, AutoTokenizer, AutoModelForCausalLM
import datasets
from peft import PeftModel
from transformers import DataCollatorWithPadding, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType
from copy import deepcopy
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="./model/Qwen2.5-1.5B")
parser.add_argument("--data_path", type=str, default="./dialogue_processed")
parser.add_argument("--per_device_train_batch_size", type=int, default=1)
parser.add_argument("--learning_rate", type=float, default=5e-5)
parser.add_argument("--num_epochs", type=int, default=10)
parser.add_argument("--output_path", type=str, default="./model/Qwen2.5-1.5B-instruct-boqi")
parser.add_argument("--device", type=str, default="cuda")
# parser.add_argument("--total_batch_size", type=int, default=32)
parser.add_argument("--ckpt_path", type=str, default="./model/Qwen2.5-1.5B-lora")


args = parser.parse_args()
 

def finetune(use_lora=False):

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
    print("model loaded")
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,  # LoRA for causal language modeling task
        r=8,  # Rank of LoRA
        lora_alpha=32,  # Alpha scaling factor for LoRA
        lora_dropout=0.1,  # Dropout rate for LoRA layers
        target_modules=["q_proj", "v_proj"],  # Apply LoRA to specific layers
    )

    dataset = datasets.load_dataset(args.data_path)
    # dataset["train"]=dataset["train"].select(range(1000))
    print(len(dataset["train"]))
    def preprocess_function(example):
        # input = f"{example['instruction']} {example['input']} {example['output']}"
        # input =f"{example['text']} {example['label']} "
        tokenized_inputs = tokenizer(
            input, 
            truncation=True, 
            padding="max_length", 
            max_length=1024,
            # return_tensors="pt",  
            # return_attention_mask=True  
        )
        
        # query=f"{example['text']}"
        # tokenized_query = tokenizer(
        #     query, 
        #     truncation=True, 
        #     padding="max_length", 
        #     max_length=1024,
        # )
        # len_input=tokenized_inputs["attention_mask"].count(1)   
        # len_query=tokenized_query["attention_mask"].count(1)
        # no_loss_list=[-100]*len_query
        # print("len_query:",len_query)
        # print("len_input:",len_input) 
        
        tokenized_inputs["labels"] = deepcopy(tokenized_inputs["input_ids"])
        
        # tokenized_inputs["labels"][:len_query] = deepcopy(no_loss_list)
        
        # one_list=[1]*len(tokenized_inputs["input_ids"])
        # tokenized_inputs["attention_mask"][:len(tokenized_inputs["input_ids"])] = deepcopy(one_list)
        return tokenized_inputs
    
    dataset["train"] = dataset["train"].map(preprocess_function, remove_columns=dataset["train"].column_names)
    
    print(len(dataset["train"]))
    print("dataset loaded")    
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    training_args = TrainingArguments(
        output_dir=args.output_path,
        do_train=True,
        eval_strategy="no", 
        save_strategy="steps",
        save_steps=2000,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        num_train_epochs=args.num_epochs,
        weight_decay=0.01,
        bf16=True,
        dataloader_num_workers=32,
    )
    if use_lora:
        model = get_peft_model(model, lora_config)
        
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        data_collator=data_collator,
        tokenizer=tokenizer, 
    )

    # Step 6: Train!
    trainer.train()


finetune(use_lora=True)


print("finetune done")