# 🤖ChatBot

## Chatbot and Virtual Human
### 🕹️Basic Chatbot
* Chatbot Execution File：`chatbot.py`

Code Execution Method：
```
python chatbot.py   --model_path $YOUR_BASE_MODEL_PATH  \
                    --ckp_path $YOUR_CHECKPOINT_PATH  \
                    --summary_model_path $YOUR_SUMMARY_MODEL_PATH\
                    --knowledge_path $YOUR_KNOWLEDGE_PATH\
                    --summary_knowledge_path $YOUR_SUMMARY_KNOWLEDGE_PATH\
```
#### ChatBot Function
- Type `\quit` to end the conversation.
- Type `\newsession` to clear the conversation history and start a new conversation.
- Type `\history` to output a summary of past conversation history.
- Type `\RAG_on` to enable external knowledge augmentation mode.
- Type `\RAG_off` to disable external knowledge augmentation mode.


### 🎸Virtual Human Chatbot
> We built the character of Ichiri Gotou from Bocchi the Rock！

![bocchi](/bocchi.jpg)


* Virtual Human Execution File: `VirtualCharacter.py`
* Code Execution Method:
```
python VirtualCharacter.py  --model_path $YOUR_BASE_MODEL_PATH  \
                            --ckp_path $YOUR_CHECKPOINT_PATH  \
```


## Instruction Fine-tuning of the LLM Base Model
> We use [Qwen-2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B)、[Qwen-2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B) as our base model.
>
> We use [alpaca-cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned) as our instruction fine-tuning dataset.
### Full-Scale Fine-tuning
* Fine-tuned Model Code File：`finetune_model.py`


Code Execution Method：
```
python finetune_model.py --model_path $YOUR_MODEL_PATH  \
                         --data_path $TRAIN_DATA_PATH  \
                         --num_epochs 3\
                         --output_path $YOUR_FINETUNED_MODEL_SAVE_PATH\
```



### LoRA Fine-tuning
* LoRA Fine-tuned Model Code File：`finetune_lora.py`

Code Execution Method：
```
python finetune_lora.py --model_path $YOUR_MODEL_PATH  \
                         --data_path $TRAIN_DATA_PATH  \
                         --num_epochs 5\
                         --output_path $YOUR_CHECKPOINT_SAVE_PATH\
```


## Model Link
[model link](https://jbox.sjtu.edu.cn/l/812Wce)


## Our RAG Frame

> To improve retrieval accuracy in the Retrieval Pool, we use ChatGPT to summarize long knowledge entries, creating
> a concise Summary Pool. We then embed both the summarized knowledge and the user's query using Qwen-1.5B with two
> embedding methods: averaging the input embeddings and using the last hidden layer before output. We employ the FAISS
> library for efficient retrieval, using vector dot product to measure similarity. The retrieved information is used
> to assist the chatbot in generating more accurate responses.

![RAG Frame](/RAG.png)


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


















