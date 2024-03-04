from transformers import LlamaForCausalLM, LlamaTokenizer, GenerationConfig
import torch
import os
import json
from tqdm import tqdm

import argparse
import torch.nn as nn
from m3apo.vcd.experiments.eval.language_dict import language_dict
import debugpy
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from m3apo.utils.utils import process_jsonl, write_jsonl
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
        
        torch.cuda.empty_cache()
        return results


def main(args):
    data_path = os.path.join(args.input_data_dir, args.data_file)
    target_dir = os.path.join(args.output_data_dir, os.path.splitext(args.data_file)[0])
    target_path = target_dir + "_{proc}_{end}.jsonl".format(proc = args.begin_index,end = args.begin_index + args.data_length)
    
    target_path = target_dir
    data = process_jsonl(data_path)
    if args.reference_style == "ref_file":
        ref_data = process_jsonl(args.reference_en_file)
    else:
        ref_data = data
    
    rm_model_base = AutoModelForSeq2SeqLM.from_pretrained(args.reward_model_path,device_map='auto')
    rm_model_base = rm_model_base.eval()
    rm_tokenizer = AutoTokenizer.from_pretrained(args.reward_model_path)


    begin_index = args.begin_index
    data_length = args.data_length
    end_index = min(len(data), begin_index + data_length)
    print("begin_index: {}, end_index: {}".format(begin_index, end_index))

    result = []
    for i in tqdm(range(begin_index,end_index)):
        item = data[i]
        ref_item = ref_data[i]
        assert item['question_id'] == ref_item['question_id']
        lang = langs[language_dict[args.language]['full_name']]
        if lang == "English":
            continue
        item_reward_list=[]
        for it in item['answer']: 
            reward_list = []
            en_answers = ref_item['answer'] if args.reference_style == "ref_file" else ref_item['en_answer']
            en_answers = en_answers if isinstance(en_answers, list) else [en_answers]
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
    parser.add_argument('-o','--output_data_dir',type=str, default="/mnt/petrelfs/songmingyang/songmingyang/runs/llava/test/add_ppl/20_en_refs/")
    parser.add_argument('-d','--data_file', type=str)
    parser.add_argument('-l','--language', type=str, default="en")
    parser.add_argument('-rs','--reference_style',type=str, default="ref_file")
    args = parser.parse_args()
    main(args)
            



