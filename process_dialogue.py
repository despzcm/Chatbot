import json
dialogues = []
TXT_PATH = r'MemoryBank\dialogue.txt'
JSON_PATH = r'MemoryBank\dialogue_processed2.json'

# with open(TXT_PATH, 'r',encoding='utf-8') as txt:
#     temp_str=''
#     has_bocchi=False
#     has_others=False
#     count=0
#     total=1538
#     texts=txt.readlines()
#     for line in texts:
#         if ':' in line:
#             name, dialogue = line.split(':', 1)
#             if '波奇' in name:
#                 has_bocchi=True
#                 if count>0 and '波奇' in texts[count-1] and len(temp_str)>0:
#                     temp_str+=dialogue
#                 else:
#                     temp_str+='assistant:'+dialogue
#                 if has_bocchi and has_others:
#                     dialogues.append({"text" : temp_str})
#                     temp_str=''
#                     has_bocchi=False
#                     has_others=False
#             else:#其他人
#                 temp_str+='user:'+dialogue
#                 has_others=True
#         else:#空行
#             if len(temp_str)>0:
#                 dialogues.append({"text" : temp_str})
#                 temp_str=''
#                 has_bocchi=False
#                 has_others=False
#         count+=1
#         print(f'Processing {count}/{total} lines')
# with open(JSON_PATH, 'w', encoding='utf-8') as json_file:
#     json.dump(dialogues, json_file, ensure_ascii=False, indent=4)
with open(TXT_PATH, 'r',encoding='utf-8') as txt:
    
    temp_str1=''
    temp_str2=''
    count=0
    texts=txt.readlines()
    for i in range(len(texts)):
        texts[i]=texts[i].split(':', 1)
    while True:
        print(count,'/',len(texts))
        if count>=len(texts):
            break
        if len(texts[count])==1:
            count+=1
            continue
        name, dialogue=texts[count]
        if '波奇' not in name:#User
            if count+1==len(texts) :
                break
            if len(texts[count+1])==1:
                count+=2
                continue
            if '波奇' in texts[count+1][0]:
                temp_str1+='<|im_start|>user\n'+(dialogue[0:-1]+'<|im_end|>\n')
                temp_str2+='<|im_start|>assistant\n'
                count+=1
                while '波奇' in texts[count][0]:
                    if count>=len(texts) or len(texts[count+1])==1:
                        temp_str2+='<|im_end|>\n'
                        dialogues.append({"text" : temp_str1,"label":temp_str2})
                        temp_str1=''
                        temp_str2=''
                        break
                    temp_str2+=(texts[count][1][0:-1]+'\n')
                    count+=1
                if len(temp_str1)>0:
                    temp_str2+='<|im_end|>\n'
                    dialogues.append({"text" : temp_str1,"label":temp_str2})
                    temp_str1=''
                    temp_str2=''
                continue
            if count+2==len(texts) :
                break
            if len(texts[count+1])==1 or len(texts[count+1])==1:
                count+=3
                continue
            if ('波奇' not in texts[count+1][0]) and ('波奇' in texts[count+2][0]):
                temp_str1+='<|im_start|>user\n'+(dialogue[0:-1]+'<|im_end|>\n')
                temp_str1+='<|im_start|>user\n'+(texts[count+1][1][0:-1]+'<|im_end|>\n')
                temp_str2+='<|im_start|>assistant\n'
                count+=2
                while '波奇' in texts[count][0]:
                    if count>=len(texts) or len(texts[count+1])==1:
                        temp_str2+='<|im_end|>\n'
                        dialogues.append({"text" : temp_str1,"label":temp_str2})
                        temp_str1=''
                        temp_str2=''
                        break
                    temp_str2+=(texts[count][1][0:-1]+'\n')
                    count+=1
                if len(temp_str1)>0:
                    temp_str2+='<|im_end|>\n'
                    dialogues.append({"text" : temp_str1,"label":temp_str2})
                    temp_str1=''
                    temp_str2=''
            count+=1
        else:
            count+=1
            

with open(JSON_PATH, 'w', encoding='utf-8') as json_file:
    json.dump(dialogues, json_file, ensure_ascii=False, indent=4)