# 🤖ChatBot

## Chatbot and Virtual Human
### Basic Chatbot
* Chatbot Execution File：`chatbot.py`

Code Execution Method：
```
python chatbot.py   --model_path $YOUR_BASE_MODEL_PATH  \
                    --ckp_path $YOUR_CHECKPOINT_PATH  \
                    --summary_model_path $YOUR_SUMMARY_MODEL_PATH\
                    --knowledge_path $YOUR_KNOWLEDGE_PATH\
                    --summary_knowledge_path $YOUR_SUMMARY_KNOWLEDGE_PATH\
```

### Virtual Human Chatbot
* We built the character of Ichiri Gotou from Bocchi the Rock！

![bocchi](/bocchi.jpg)


* Virtual Human Execution File: `VirtualCharacter.py`
* Code Execution Method:
```
python VirtualCharacter.py  --model_path $YOUR_BASE_MODEL_PATH  \
                            --ckp_path $YOUR_CHECKPOINT_PATH  \
```


## Instruction Fine-tuning of the LLM Base Model
### Full-Scale Fine-tuning
* Fine-tuned Model Code File：`finetune_model.py`
* We use [alpaca-cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned)as our instruction fine-tuning dataset.

Code Execution Method：
```
python finetune_model.py --model_path $YOUR_MODEL_PATH  \
                         --data_path $TRAIN_DATA_PATH  \
                         --num_epochs 3\
                         --output_path $YOUR_FINETUNED_MODEL_SAVE_PATH\
```



### LoRA Fine-tuning
* LoRA Fine-tuned Model Code File：`finetune_lora.py`
* We use [alpaca-cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned)as our instruction fine-tuning dataset.

Code Execution Method：
```
python finetune_lora.py --model_path $YOUR_MODEL_PATH  \
                         --data_path $TRAIN_DATA_PATH  \
                         --num_epochs 5\
                         --output_path $YOUR_CHECKPOINT_SAVE_PATH\
```


## Additional Files
* `RAG.py` External Knowledge Augmentation
* `corpus_process.py` Simple Corpus Segmentation
* `history_sumary.py` History Information Summarization
* `knowledge.json` External Knowledge Base
* `knowledge_summary.json` External Summarized Knowledge Base

* `dialogue.txt` Raw Corpus for Virtual Human
* `corpus_seg.json` Simple Segmented Training Corpus
* `dialogue_processed.json` Formatted Corpus for Virtual Human
* `process_dialogue.py` Virtual Human Corpus Formatting Script



## Model Link
[model link](https://jbox.sjtu.edu.cn/l/812Wce)

















