#-*- coding:utf-8 -*-
import argparse
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import sys
import torch
from transformers import TrainingArguments, HfArgumentParser, Trainer, AutoTokenizer, AutoModelForCausalLM ,DataCollatorWithPadding,DataCollatorForLanguageModeling,TextIteratorStreamer,pipeline
import datasets
from peft import LoraConfig, get_peft_model, TaskType,PeftModel
from threading import Thread
from history_sumary import HistorySummary
from RAG import RAG_search
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="./model/Qwen2.5-1.5B")
parser.add_argument("--ckp_path", type=str, default="./Qwen2.5-1.5B-lora-v2/checkpoint-64700")
parser.add_argument("--torch_dtype", type=str, default="bfloat16")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--sumary_model_path", type=str, default="./Qwen2.5-1.5B-instruct")
parser.add_argument("--knowledge_path", type=str, default="./retrieval_knowledge.jsonl")
parser.add_argument("--summary_knowledge_path",type=str,default='./knowledge_summary.json')
args = parser.parse_args()


class Chatbot:
    def __init__(self,model_name_or_path:str,
                 torch_dtype:str="bfloat16",
                 device:str='cuda:0',
                 lora_ckp_path:str=None,
                 sumary_model_path:str=None,
                 summary_knowledge_list:str=None,
                 knowledge_list:str=None
                 ):
        if device == "cuda:0" and not torch.cuda.is_available():
            device = "cpu"
            
        self.model_name=model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=getattr(torch, torch_dtype))
        if lora_ckp_path is not None:
            self.model = PeftModel.from_pretrained(self.model, lora_ckp_path)
        self.model.to(device)
        
        self.history = []
        self.HistorySummary = HistorySummary(model_path=sumary_model_path)
        self.encoded_history = None
        self.history_summary=''
        
        self.RAG_search=RAG_search()
        self.RAG_flag=False
        self.knowledge_list=knowledge_list
        self.summary_knowledge_list=summary_knowledge_list
        

    def get_fromatted_input(self,text_dict_list:List[Dict[str,str]],add_generation_prompt:bool=False):
        text=''
        for text_dict in text_dict_list:
            text+=''+text_dict['role']+':'+text_dict['content']+'\n'
        if add_generation_prompt:
            text+='assistant:'
        return text
    
    def get_fromatted_input_withSummary(self,text_dict_list:List[Dict[str,str]]):
        text=''
        for text_dict in text_dict_list:
            if text_dict['role']=="system":
                text+=''+text_dict['role']+':'+text_dict['content']+'\n'
                break
        # print(self.history_summary)
        if self.history_summary!='':
            # text+="The following is the previous conversation summary:\n"
            text+=self.history_summary
            text+="\n"
            # text+='As the chatbot, you need to use the above memory to assist user\'s query as following.\n'  
            # print(111111111111111111111111111)
        return text
    
    def sumary_history(self):
        text=""
        for text_dict in self.history:
            if text_dict['role']=="system":
                continue
            text+=f"{text_dict['role']}: {text_dict['content']}\n"
        if text!="":
            self.history_summary=self.HistorySummary.summarize_history(text)
    def remove_user_segment(self,input_string):
        user_index = input_string.find("user:")  
        if user_index != -1:
            return input_string[:user_index]
        else:
            user_index = input_string.find("User:")
            user_index2=input_string.find("用户：")
            user_index3=input_string.find("用户:")
            if user_index != -1:
                return input_string[:user_index]
            elif user_index2 != -1:
                return input_string[:user_index2]
            elif user_index3 != -1:
                return input_string[:user_index3]
            else:
                return input_string
                 
    def start_chat(self,
                   max_length_single_chat:int=2048,
                   content_length:int=8192,
                   system_order:str=None,
                   ):
        
        while True:
            if len(self.history)==0:
                print("Qwen standing by. Please type your content. \n Type '\quit' to stop chat.\n Type '\\newsession' to clear history")
                print(" Type '\\history' to show history summary")
                print(" Type '\\RAG_on' to enable RAG")
                print(" Type '\\RAG_off' to disable RAG")
                if system_order is None or system_order=='':
                    system_order = "You are Qwen2.5, a wise rational robot helper.You should offer help to users."
                self.history.append(
                    {"role": "system",
                    "content": system_order}
                    )
                text=self.get_fromatted_input(self.history)
                self.encoded_history = self.tokenizer(text, return_tensors='pt').to(self.model.device)
            input_text = input("User: ")
            if input_text == r"\quit":
                print('Chat Terminated')
                break
            elif input_text == r"\newsession":
                print('History Cleared')
                self.history = []
                self.encoded_history = None
                self.history_summary=''
                self.RAG_flag=False
                continue
            elif input_text == r"\history":
                self.sumary_history()
                print("\n=== Dialogue History Summary ===\n")
                # print(self.tokenizer.batch_decode(self.encoded_history['input_ids'],skip_special_tokens=False)[0])
                print(self.history_summary)
                print("==============================\n")
            elif input_text == r"\RAG_on":    
                self.RAG_flag=True
                print("RAG enabled")
            elif input_text == r"\RAG_off":
                self.RAG_flag=False
                print("RAG disabled")
            else:
                self.history.append(
                    {"role": "user","content": input_text}
                    )
                if self.RAG_flag:
                    most_similar_knowledge = self.RAG_search.find_most_similar_info(knowledge_lib, input_text, embedding_type='input', top_k=1,knowledge_list=knowledge_list)
                    input_text=most_similar_knowledge+'\n请根据上面的信息回答问题'+input_text
                
                text=self.get_fromatted_input([{"role": "user","content": input_text}],True)
                # print("--------------------User: "+input_text)
                self.encoded_history = {
                    'input_ids': torch.cat([self.encoded_history['input_ids'], 
                                                  self.tokenizer(text, return_tensors='pt').to(self.model.device)['input_ids']], 
                                                 dim=1),
                    'attention_mask': torch.cat([self.encoded_history['attention_mask'], 
                                                  self.tokenizer(text, return_tensors='pt').to(self.model.device)['attention_mask']],
                                                 dim=1),
                                        }
                
                
                generate_ids = self.model.generate(**self.encoded_history,
                                            max_new_tokens=max_length_single_chat,
                                            temperature=0.2,
                                            num_return_sequences=1,
                                            top_k=20,
                                            top_p=0.8,
                                            do_sample=True,
                                            no_repeat_ngram_size=2,
                                            repetition_penalty=1.2,
                                            )
                generate_ids=[output_ids[len(input_ids):] for input_ids, output_ids in zip(self.encoded_history['input_ids'], generate_ids)]

                
                new_text=generate_ids[0]
                if new_text[-1]==151643:
                    new_text=new_text[:-1]
                    #151645
                new_text=torch.cat((new_text,torch.tensor([198]).to(self.model.device)))[None,:]
                ##history加入bot对话
                self.encoded_history={
                    'input_ids':torch.cat([self.encoded_history['input_ids'],new_text],dim=1),
                    'attention_mask':torch.cat([self.encoded_history['attention_mask'],torch.ones_like(new_text).to(self.model.device)],dim=1)
                }
                response = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0]
                response=self.remove_user_segment(response) 
                self.history.append(
                    {"role": "assistant",
                    "content": response}
                    )
                ##超长上下文清除
                if self.encoded_history['input_ids'].shape[1]>content_length:
                    self.encoded_history['input_ids']=self.encoded_history['input_ids'][:,-content_length:]
                    self.encoded_history['attention_mask']=self.encoded_history['attention_mask'][:,-content_length:]
                
                print("Qwen: "+response)
                
    def stream_container(self,streamer):
        for new_text in streamer:
            yield new_text
    def start_stream_chat(self
                          ,max_length_single_chat:int=512
                          ,system_order:str=None
                          ,content_length:int=4096):

        streamer=TextIteratorStreamer(self.tokenizer,skip_prompt=True,skip_special_tokens=True)
        if system_order is None or system_order=='':
            system_order = "You are Qwen2.5, a wise rational robot helper.You should offer help to users."
        while True:
            
            if len(self.history)==0:
                print("Qwen standing by. Please type your content. \n Type '\quit' to stop chat. \n Type '\\newsession' to clear history")
                self.history.append(
                    {"role": "system",
                    "content": system_order}
                    )
                text=self.get_fromatted_input(self.history)
                self.encoded_history = self.tokenizer(text, return_tensors='pt').to(self.model.device)
            input_text = input("User: ")
            if input_text == r"\quit":
                print('Chat Terminated')
                break
            elif input_text == r"\newsession":
                print('History Cleared')
                self.history = []
                self.encoded_history = None
                continue
            elif input_text == r"\history":
                print("\n=== Encoded Dialogue History ===")
                print(self.tokenizer.batch_decode(self.encoded_history['input_ids'],skip_special_tokens=False)[0])
                print("========================\n")
            else:#input text
                self.history.append(
                    {"role": "user","content": input_text}
                    )
                text=self.get_fromatted_input([{"role": "user","content": input_text}],True)

                #history加入用户对话
                self.encoded_history = {
                    'input_ids': torch.cat([self.encoded_history['input_ids'], 
                                                  self.tokenizer(text, return_tensors='pt').to(self.model.device)['input_ids']], 
                                                 dim=1),
                    'attention_mask': torch.cat([self.encoded_history['attention_mask'], 
                                                  self.tokenizer(text, return_tensors='pt').to(self.model.device)['attention_mask']],
                                                 dim=1),
                                        }
                
                
                generation_kwargs=dict(
                    input_ids=self.encoded_history['input_ids'],
                    attention_mask=self.encoded_history['attention_mask'],
                    max_new_tokens=max_length_single_chat,
                    temperature=0.5,
                    num_return_sequences=1,
                    top_k=50,
                    top_p=0.95,
                    do_sample=True,
                    no_repeat_ngram_size=2,
                    repetition_penalty=1.2,
                    streamer=streamer
                )
                thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
                thread.start()
                text_generator = self.stream_container(streamer)
                print("Qwen: ",end='')
                
                ##输出信息
                output_text = ''
                for new_text in text_generator:
                    print(new_text,end='')
                    output_text+=new_text
                print('\n')
                
                ##history更新
                self.history.append(
                    {"role": "assistant",
                    "content": output_text}
                    )
                #151645
                re_tokenized_text=self.tokenizer(output_text,return_tensors='pt').to(self.model.device)['input_ids']
                re_tokenized_text=torch.cat((re_tokenized_text,torch.tensor([[198]]).to(self.model.device)),dim=1)
                
                self.encoded_history={
                    'input_ids':torch.cat([self.encoded_history['input_ids'],re_tokenized_text],dim=1),
                    'attention_mask':torch.cat([self.encoded_history['attention_mask'],torch.ones_like(re_tokenized_text)],dim=1)
                }
                
                ##超长上下文清除
                if self.encoded_history['input_ids'].shape[1]>content_length:
                    self.encoded_history['input_ids']=self.encoded_history['input_ids'][:,-content_length:]
                    self.encoded_history['attention_mask']=self.encoded_history['attention_mask'][:,-content_length:]
                    

if __name__ == '__main__':
    knowledge_lib=[]
    knowledge_list=[]
    with open(args.summary_knowledge_path, 'r') as f:
        knowledge_lib = json.load(f)
    
    with open(args.knowledge_path, 'r') as f:
        knowledge_list = json.load(f)
        
    print("loading knowledge lib successfully")
    
    chatbot = Chatbot(
        model_name_or_path=args.model_path,
        torch_dtype=args.torch_dtype,
        device=args.device,
        lora_ckp_path=args.ckp_path,
        sumary_model_path=args.sumary_model_path,
        summary_knowledge_list=knowledge_lib,
        knowledge_list=knowledge_list                               
    )
    chatbot.start_chat(
        system_order=
        '''''',
        
    )
    # chatbot.start_stream_chat(
    #     system_order=
    #     '''''',
    # )
