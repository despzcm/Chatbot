import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import time
import faiss
import json

class RAG_search:
    def __init__(self, model_path='./model/Qwen2.5-1.5B'):

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            add_eos_token=False, 
        )
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.embedding_have_been_calculated = False
        self.all_embeddings=None
        self.embedding_type = None
        self.index=None

    def get_embeddingByOutput(self, sentence):
        inputs = self.tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)

        with torch.no_grad():
            outputs = self.model(**inputs)

        last_hidden_state = outputs.logits  

        embeddings = last_hidden_state[0, :, :].numpy() 
        sentence_embedding = np.mean(embeddings, axis=0) 
        return sentence_embedding
    
    def get_embeddingByInput(self, sentence):
        inputs = self.tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)

        with torch.no_grad():
            embed_tokens = self.model.get_input_embeddings()(inputs['input_ids'])  
        sentence_embedding = embed_tokens.numpy().mean(axis=1)
        return sentence_embedding
    
    def get_ALL_embeddings(self, steps, embedding_type):
        if self.embedding_have_been_calculated:
            return
        
        if embedding_type == 'input':
            self.all_embeddings = [self.get_embeddingByInput(step) for step in steps]
        else:
            self.all_embeddings= [self.get_embeddingByOutput(step) for step in steps]
        self.embedding_have_been_calculated = True
        return
    
    
    def find_most_similar_info(self, knowledge_lib, query,embedding_type='output',top_k=3,knowledge_list=None):
        if not self.embedding_have_been_calculated:
            self.get_ALL_embeddings(knowledge_lib, embedding_type)
        if not self.embedding_type:
            self.embedding_type = embedding_type
        else:
            if self.embedding_type != embedding_type:
                self.embedding_have_been_calculated = False
                self.get_ALL_embeddings(knowledge_lib, embedding_type)
                self.embedding_type = embedding_type
                # print("Embedding type has been changed, recalculate all embeddings")
            
        
        if embedding_type == 'input':
            query_embedding = self.get_embeddingByInput(query)
        else:
            query_embedding = self.get_embeddingByOutput(query)

        all_embeddings = np.vstack(self.all_embeddings)
        
        self.index, pca=self.construct_faiss_index(all_embeddings)
        
        most_similar_steps=self.search_with_faiss(query_embedding, knowledge_list, self.index, k=top_k,pca=pca)
        return most_similar_steps
    
    def construct_faiss_index(self,embeddings):
        pca = faiss.PCAMatrix(embeddings.shape[-1], 32)
        d = embeddings.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(embeddings)
        return index, pca
    
    def search_with_faiss(self,query_embed, data_list, index, k=1,pca=None):
        if self.embedding_type == 'output':
            query_embed=np.expand_dims(query_embed, axis=0)
        IP, indices = index.search(query_embed, k)

        prompt_str_list = [str(data_list[idx.item()]) for idx in indices[0][:k]]
        prompt = '\n'.join(prompt_str_list)
        return prompt

    

if __name__ == "__main__":
    
    query="吴伯通的官职经历和成就有哪些?"

    # knowledge_path='./retrieval_knowledge.jsonl'
    # summary_knowledge_path='./knowledge_summary.json'
    # with open(summary_knowledge_path, 'r') as f:
    #     knowledge_lib = json.load(f)
    
    # with open(knowledge_path, 'r') as f:
    #     knowledge_list = json.load(f)
    # print("loading knowledge lib successfully")
    
     
    # cur_time = time.time()
    # rag_search = RAG_search()
    # most_similar_knowledge = rag_search.find_most_similar_info(knowledge_lib, query, embedding_type='input', top_k=1,knowledge_list=knowledge_list)
    # print(f"最相似的知识:\n {most_similar_knowledge}")
    # end_time = time.time()
    # print(f"Time: {end_time - cur_time}")
