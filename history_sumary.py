import argparse
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import sys
import os
import torch
from transformers import TrainingArguments, HfArgumentParser, Trainer, AutoTokenizer, AutoModelForCausalLM
import datasets
import transformers
from datasets import load_dataset, DatasetDict


parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="./model/Qwen2.5-1.5B-instruct")
args = parser.parse_args()


class HistorySummary:
    def __init__(self, model_path: str, max_context_length: int = 200):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model =AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16,)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.max_context_length = max_context_length
        self.dialog_history = []

    def summarize_history(self,text):
        # print("---------text:",text)
        messages = [
            {"role": "system", "content": "Please provide a concise summary of the following conversation. You have to retain key information:"},
            {"role": "user", "content": '[conversation]\n'+text}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # print("---------response:",response)
        return response




if __name__ == "__main__":
    chatbot = HistorySummary(model_path=args.model_path)
    text='''
            User: I like science fiction novels. Can you recommend a famous one to me?
            ChatBot: 1984 by George Orwell is considered the most influential novel of all time, it's about totalitarianism.
            What do you think?
            User: What type of novels do I like
            ChatBot: Science Fiction
            Can we talk more on this topic?
        '''
    chatbot.summarize_history(text)
