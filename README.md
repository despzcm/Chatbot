# 🤖ChatBot

## 聊天机器人与虚拟人
### 基础聊天机器人
* 聊天机器人运行文件：chatbot.py

代码运行方法：
```
python chatbot.py   --model_path $YOUR_BASE_MODEL_PATH  \
                    --ckp_path $YOUR_CHECKPOINT_PATH  \
                    --summary_model_path $YOUR_SUMMARY_MODEL_PATH\
                    --knowledge_path $YOUR_KNOWLEDGE_PATH\
                    --summary_knowledge_path $YOUR_SUMMARY_KNOWLEDGE_PATH\
```

### “虚拟人”聊天机器人
* 我们构建了《孤独摇滚》中的后藤一里
* 虚拟人运行文件: VirtualCharacter.py
* 代码运行方法:
```
python VirtualCharacter.py  --model_path $YOUR_BASE_MODEL_PATH  \
                            --ckp_path $YOUR_CHECKPOINT_PATH  \
```


## LLM基底模型的指令微调
### 全量微调
* 微调模型代码文件：finetune_model.py
* 我们使用[alpaca-cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned)作为我们的指令微调数据集

代码运行方法：
```
python finetune_model.py --model_path $YOUR_MODEL_PATH  \
                         --data_path $TRAIN_DATA_PATH  \
                         --num_epochs 3\
                         --output_path $YOUR_FINETUNED_MODEL_SAVE_PATH\
```

模型链接：[model link](https://jbox.sjtu.edu.cn/l/812Wce)

### LoRA微调
* LoRA微调模型代码文件：finetune_lora.py
* 我们使用[alpaca-cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned)作为我们的指令微调数据集

代码运行方法：
```
python finetune_lora.py --model_path $YOUR_MODEL_PATH  \
                         --data_path $TRAIN_DATA_PATH  \
                         --num_epochs 5\
                         --output_path $YOUR_CHECKPOINT_SAVE_PATH\
```


## 附加文件
* `RAG.py` 外部知识增加
* `corpus_process.py` 简单语料分割
* `history_sumary.py` 历史信息总结
* `knowledge.json` 外部知识库
* `knowledge_summary.json` 外部总结知识库

* `dialogue.txt` 虚拟人原始语料
* `corpus_seg.json` 简单分割训练语料
* `dialogue_processed.json` 虚拟人格式化语料
* `process_dialogue.py` 虚拟人语料格式化脚本



## 模型链接：
[model link](https://jbox.sjtu.edu.cn/l/812Wce)
* `Qwen2.5-1.5B-lora` LoRA训练模型checkpoints
* `Qwen2.5-1.5B-Instruct-bocchi3` 虚拟人模型checkpoints
















