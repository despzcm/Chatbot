import json


with open('./corpus.txt', 'r', encoding='utf-8') as file:
    content = file.read()


chunk_size = 200  
chunks = []  
current_chunk = "" 

i = 0
while i < len(content):
    remaining_text = content[i:i+chunk_size]
    
    if len(remaining_text) == chunk_size:
        if '\n' in remaining_text:
            newline_pos = remaining_text.rfind('\n') 
            current_chunk += remaining_text[:newline_pos+1]  
            chunks.append({"text": current_chunk.strip()})
            current_chunk = ""  
            i += newline_pos + 1 
        else:
            
            current_chunk += remaining_text
            chunks.append({"text": current_chunk.strip()})
            current_chunk = "" 
            i += chunk_size
    else:
        
        current_chunk += remaining_text
        if current_chunk.strip(): 
            chunks.append({"text": current_chunk.strip()})
        break  


with open('./NLP/corpus_seg.json', 'w', encoding='utf-8') as json_file:
    json.dump(chunks, json_file, ensure_ascii=False, indent=4)

print("data segmentation completed!")
