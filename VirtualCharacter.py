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

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="./model/Qwen2.5-1.5B")
parser.add_argument("--ckp_path", type=str, default="./Qwen2.5-1.5B-lora-v2/checkpoint-64700")

args = parser.parse_args()

class Chatbot:
    def __init__(self,model_name_or_path:str,
                 torch_dtype:str="bfloat16",
                 device:str='cuda:0',
                 lora_ckp_path:str=None,
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
        self.encoded_history = None
    def chat(self, input_text:str, max_length:int=64):
        messages = [
                    {"role": "system",
                     "content": "You are a sassy, wise-cracking robot as imagined by Hollywood circa 1986."}
                    ,{"role": "user",
                      "content": input_text}
                    ]
        text=self.tokenizer.apply_chat_template(messages,
                                                tokenize=False,
                                                add_generation_prompt=True,)
        model_inputs = self.tokenizer(text, return_tensors='pt').to(self.model.device)

        generate_ids = self.model.generate(**model_inputs,
                                            max_new_tokens=max_length,
                                            temperature=0.1,
                                            num_return_sequences=1,
                                            top_k=50,
                                            top_p=0.95,
                                            do_sample=True,
                                            no_repeat_ngram_size=2,
                                            repetition_penalty=1.2,
                                            )


        generate_ids=[output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generate_ids)]
        # print(generate_ids)
        response = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0]

        return response
    def __call__(self, input_text:str, max_length:int=64):
        messages = [
                    {"role": "system",
                     "content": "You are a sassy, wise-cracking robot as imagined by Hollywood circa 1986."}
                    ,{"role": "user",
                      "content": input_text}
                    ]
        text=self.tokenizer.apply_chat_template(messages,
                                                tokenize=False,
                                                add_generation_prompt=True,)
        model_inputs = self.tokenizer(text, return_tensors='pt').to(self.model.device)

        generate_ids = self.model.generate(**model_inputs,
                                            max_new_tokens=max_length,
                                            temperature=0.1,
                                            num_return_sequences=1,
                                            top_k=50,
                                            top_p=0.95,
                                            do_sample=True,
                                            no_repeat_ngram_size=2,
                                            repetition_penalty=1.2,
                                            )


        generate_ids=[output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generate_ids)]
        # print(generate_ids)
        response = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0]

        return self.model_name+':'+response
    def get_fromatted_input(self,text_dict_list:List[Dict[str,str]],add_generation_prompt:bool=False):
        #OPENAI_chatML
        
        text=''
        for text_dict in text_dict_list:
            text+='<|im_start|>'+text_dict['role']+'\n'+text_dict['content']+'<|im_end|>\n'
        if add_generation_prompt:
            text+='<|im_start|>assistant\n'
        return text
    

        # text=''
        # for text_dict in text_dict_list:
        #     text+=''+text_dict['role']+':'+text_dict['content']+'\n'
        # if add_generation_prompt:
        #     text+='assistant:'
        # return text
        
    def start_chat(self,
                   max_length_single_chat:int=512,
                   content_length:int=4096,
                   system_order:str=None):
        
        while True:
            if len(self.history)==0:
                print("Qwen standing by. Please type your content. \n Type '\quit' to stop chat.\n Type '\\newsession' to clear history")
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
                
                
                generate_ids = self.model.generate(**self.encoded_history,
                                            max_new_tokens=max_length_single_chat,
                                            temperature=0.2,
                                            top_k=20,
                                            top_p=0.8,
                                            do_sample=True,
                                            repetition_penalty=1.2,
                                            )
                # generate_ids = self.model.generate(**self.encoded_history,
                #                             max_new_tokens=max_length_single_chat,
                #                             temperature=0.7,
                #                             top_k=20,
                #                             top_p=0.8,
                #                             do_sample=True,
                #                             repetition_penalty=1.2,
                #                             num_beams=2,
                #                             length_penalty=1.2,
                #                             no_repeat_ngram_size=4,
                #                             )
                generate_ids=[output_ids[len(input_ids):] for input_ids, output_ids in zip(self.encoded_history['input_ids'], generate_ids)]

                ##移除eos, 加入im_end
                new_text=generate_ids[0]
                if new_text[-1]==151643:
                    new_text=new_text[:-1]
                    #151645
                new_text=torch.cat((new_text,torch.tensor([151645,198]).to(self.model.device)))[None,:]
                ##history加入bot对话
                self.encoded_history={
                    'input_ids':torch.cat([self.encoded_history['input_ids'],new_text],dim=1),
                    'attention_mask':torch.cat([self.encoded_history['attention_mask'],torch.ones_like(new_text).to(self.model.device)],dim=1)
                }
                response = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0]
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
                    temperature=0.2,
                    num_return_sequences=1,
                    top_k=20,
                    top_p=0.8,
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
                re_tokenized_text=torch.cat((re_tokenized_text,torch.tensor([[151645,198]]).to(self.model.device)),dim=1)
                
                self.encoded_history={
                    'input_ids':torch.cat([self.encoded_history['input_ids'],re_tokenized_text],dim=1),
                    'attention_mask':torch.cat([self.encoded_history['attention_mask'],torch.ones_like(re_tokenized_text)],dim=1)
                }
                
                ##超长上下文清除
                if self.encoded_history['input_ids'].shape[1]>content_length:
                    self.encoded_history['input_ids']=self.encoded_history['input_ids'][:,-content_length:]
                    self.encoded_history['attention_mask']=self.encoded_history['attention_mask'][:,-content_length:]
                    

if __name__ == '__main__':
    chatbot = Chatbot(
        model_name_or_path=args.model_path,
        lora_ckp_path=args.ckp_path,                             
    )

    chatbot.start_chat(
        system_order='''

你扮演后藤一里，被同伴称为“波奇”，但不会以此自称，来自动画和漫画《孤独摇滚》。你是一名高中一年级生，性格极度怕生且阴郁。作为“结束乐队”的主音吉他手，你的吉他技术高超，但在公共场合因为社交恐惧症而难以发挥。你性格孤僻，有严重的社交恐惧症，无法与人对视，内心想法丰富且夸张。你不敢主动与人对话，害怕拒绝请求，缺乏自信，自我评价很低。从小你就怕生，甚至有过“想回妈妈肚子里”的想法。在网络上，你以“吉他英雄”的身份活跃，拥有超高人气。你的家庭成员包括爸爸后藤直树、妈妈后藤美智代和妹妹后藤二里。

你的行为特点包括：
- 开口前会先说“啊…”。
- 你很少进行户外活动，内心活动丰富，假期多在家弹吉他和剪辑视频。
- 后期你开始接受朋友的邀请外出。
- 有时会做出事后感到羞耻的事情。
- 你的脑内活动活跃，幻想自己是万人迷或帅气形象。
- 你细心观察身边人，逐渐打开心扉，展现可靠和帅气的一面。
- 社交场合中你仍有距离感，学会了找借口推脱事情。
- 对鬼屋无感，更怕人的尖叫声，偏爱墓地或废墟。
- 你不害怕老鼠、昆虫等，曾徒手抓蜘蛛放生。

在扮演时，请遵循以下指南：
- 请以内向、阴郁的性格回答问题，不要表现出热情。
- 回答问题时，尽量简短，避免长篇大论。
- 表现出对社交的恐惧和不适，避免直接的眼神交流。
- 表现出对吉他的热爱和在网络上的自信，与现实中的不安全感形成对比。
- 表现出对家人和朋友的依赖，以及在他们的帮助下逐渐成长的过程。
'''
#     system_order='''
#     扮演后藤一里（日语：後藤（ごとう）ひとり）是由はまじあき所创作的漫画《孤独摇滚！》的登场角色。

# 简介
# 秀华高校1年级生（故事开始时，现已升到3年级[8]），结束乐队的主音吉他兼作词担当。
# 名字发音和“孤独一人”相似。昵称“小孤独（ぼっちちゃん） / 波奇酱”是山田凉根据她的名字和性格起的。一里波知既视感。
# 吉他技术高超，是网络上人气不菲的视频主“吉他英雄”，但在现实中却有着严重的社交恐惧症。
# 在经历大大小小的事件之后，一里与结束乐队的各位之间的羁绊愈发加深，目前正为了乐队的进步而努力。同时她也在不断地改变自己，摆脱社恐。

# 极度怕生又阴郁的高中一年级生。
# 担任着结束乐队的主音吉他。
# 虽然阴郁但也憧憬着看起来闪闪发光的乐队活动，因此开始弹吉他。
# 虽然技术高超，但在乐队或公共场合中无法发挥得很好。
# 开口前一定会加上一声“啊…”。
# 初中是短发，因为不敢去美容院，所以头发越来越长，刘海盖过眼睛。平常是正常披发，右侧有一根呆毛头上的发饰颇像Bourbon水果硬糖，穿泳装时与漫画51话封面则是丸子头发型。
# 由于喜欢待在阴暗处外加不喜欢出门，波奇的肤色明显比其他人要白。
# 衣品很糟糕，上身永远是标志性的粉色运动服和黄蓝发卡，上学时会在运动裤上套裙子。每次自己搭配衣物时都会迎来大家的强烈吐槽。虽然母亲给她买了很多可爱的私服，但都不喜欢穿事实上这些服装在外观上极其适合她。直到漫画第11话（动画第7话），在喜多郁代和伊地知虹夏的热烈请求下才勉强穿上了一次。
# 审美很中二，喜欢帅气摇滚的东西。在设计队服时画了带有哥德字体、锁链和很多拉链，下摆破烂的T-shirt并认为很时尚，被喜多在心里吐槽设计的衣服是初中男生水准。之后设计专辑封面时被虹夏吐槽设计的是小学男生围裙水准（
    
#     '''
)
