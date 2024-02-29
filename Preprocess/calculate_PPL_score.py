from transformers import LlamaForCausalLM, LlamaTokenizer, GenerationConfig
import torch
import os
import json
from tqdm import tqdm

import argparse
import torch.nn as nn
from experiments.eval.language_dict import language_dict
import debugpy
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

loss_fn = nn.CrossEntropyLoss(reduction='none')
langs = {
            'Swahili' :'swh_Latn',
            'Chinese' : "zho_Hans",
            "Bengali" : "ben_Beng",
            "German" : "deu_Latn",
            "Spanish" : "spa_Latn",
            "French" : "fra_Latn",
            "Japanese" : "jpn_Hani",
            "Russian" : "rus_Cyrl",
            "Thai" : "tha_Thai",
            "English" : "eng_Latn"
            } 

def MultiLigual_Alighment_reward_fuction(tokenizer,rm_model,outputs,labels=None,language='Chinese'):
        model = rm_model
        target_lang = 'eng_Latn'
        tokenizer.src_lang = language
        x = tokenizer(outputs, return_tensors='pt', padding='longest', truncation=True,max_length=512).to(model.device)
        tokenizer.src_lang = target_lang
        y = tokenizer(labels, return_tensors='pt', padding='longest', truncation=True,max_length=512).to(model.device)
        y.input_ids[y.input_ids == tokenizer.pad_token_id] = -100
        results = []
        
        
        
        with torch.no_grad():
            output = model(**x, labels=y.input_ids)
            loss = output.loss

            for i in range(output.logits.size(0)):
                pre = output.logits[i]
                lab = y.input_ids[i]
                result = loss_fn(pre.view(-1, output.logits.size(-1)), lab.view(-1)).mean().cpu().detach().numpy().tolist()
                results.append(1/result)
                
        # debugpy.listen(("0.0.0.0", 5678))
        # debugpy.wait_for_client()
        # breakpoint()
        
        torch.cuda.empty_cache()
        return results


def main(args):
    data_path = os.path.join(args.input_data_dir, args.data_file)
    target_dir = os.path.join(args.output_data_dir, args.data_file)
    target_path = target_dir + "_{proc}_{end}.jsonl".format(proc = args.begin_index,end = args.begin_index + args.data_length)
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
         for line in f:
            json_obj = json.loads(line)
            data.append(json_obj)

    
    rm_model_base = AutoModelForSeq2SeqLM.from_pretrained(args.reward_model_path,device_map='auto')
    rm_model_base = rm_model_base.eval()
    rm_tokenizer = AutoTokenizer.from_pretrained(args.reward_model_path)


    begin_index = args.begin_index
    data_length = args.data_length
    end_index = min(len(data), begin_index + data_length)
    print("begin_index: {}, end_index: {}".format(begin_index, end_index))

    result = []
    for i in tqdm(range(begin_index, end_index)):
        item = data[i]
        lang = langs[language_dict[args.language]['full_name']]
        if lang == "English":
            continue
        if len(item['en_answer']) == 0:
            continue
        item_reward_list=[]
        for it in item['answer']: 
            reward_list = []
            en_answers = item['en_answer'] if isinstance(item['en_answer'], list) else [item['en_answer']]
            input_answer = [it for _ in en_answers]
            output = [j for j in en_answers]
            if len(input_answer) <= 20:
                reward_list = MultiLigual_Alighment_reward_fuction(rm_tokenizer,rm_model_base,input_answer,output,lang)
            else:
                reward_list1 = MultiLigual_Alighment_reward_fuction(rm_tokenizer,rm_model_base,input_answer[:20],output[:20],lang)
                reward_list2 = MultiLigual_Alighment_reward_fuction(rm_tokenizer,rm_model_base,input_answer[20:],output[20:],lang)
                reward_list = reward_list1 + reward_list2
                assert len(reward_list) == len(output)
            reward_res = {
                'nllb-200-distilled-600M-reward-mean':sum(reward_list)/len(reward_list),
                'nllb-200-distilled-600M-reward-max':max(reward_list),
                'nllb-200-distilled-600M-reawrdlist':reward_list,
            }
            item_reward_list.append(reward_res)
            torch.cuda.empty_cache()
            # it['nllb-200-distilled-600M-reward-mean'] =  sum(reward_list)/len(reward_list)
            # it['nllb-200-distilled-600M-reward-max'] =  max(reward_list)
            # it['nllb-200-distilled-600M-reawrdlist'] =  reward_list
        item["reward_list"] = item_reward_list
        result.append(item)

    with open(target_path, 'w', encoding='utf-8') as fw:
        for item in result:
            line = json.dumps(item, ensure_ascii=False)
            fw.write(line + '\n')
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="sampling argument")
    parser.add_argument('--begin_index', type=int,default=0)
    parser.add_argument('--data_length', type=int,default=2000)
    parser.add_argument('-rm','--reward_model_path', type=str,default='/mnt/petrelfs/songmingyang/songmingyang/model/others/nllb-200-distilled-600M')
    parser.add_argument('-rf','--reference_en_file',type=str, default='/mnt/petrelfs/songmingyang/songmingyang/runs/llava/test/generations/llava_7b_v1_generation_num20_en.json')
    parser.add_argument('-i','--input_data_dir',type=str, default="/mnt/petrelfs/songmingyang/songmingyang/runs/llava/test/generations")
    parser.add_argument('-o','--output_data_dir',type=str, default="/mnt/petrelfs/songmingyang/songmingyang/runs/llava/test/add_ppl")
    parser.add_argument('-d','--data_file', type=str)
    parser.add_argument('-l','--language', type=str, default="en")
    args = parser.parse_args()
    main(args)
            



