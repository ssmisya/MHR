from transformers import LlamaForCausalLM, LlamaTokenizer, GenerationConfig
import torch
import os
import json
from tqdm import tqdm

import argparse
import torch.nn as nn
from m3apo.vcd.experiments.eval.language_dict import language_dict,nllb_200_distilled_600M_language_dict
import debugpy
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sacrebleu import corpus_bleu
from sacrebleu.metrics import BLEU, CHRF, TER

from m3apo.utils.utils import process_jsonl, write_jsonl
loss_fn = nn.CrossEntropyLoss(reduction='none')

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
    
def BLEU_Alighment_reward_fuction(tokenizer,rm_model,outputs,labels=None,language='Chinese'):
    # model = rm_model
    # target_lang = 'eng_Latn'
    # tokenizer.source_lang = language
    # tokenizer.target_lang = target_lang
    # input = tokenizer(outputs, return_tensors='pt', padding='longest', truncation=True,max_length=512).to(model.device)
    # output = model.generate(**input, forced_bos_token_id=tokenizer.lang_code_to_id[target_lang])
    # output_text = tokenizer.batch_decode(output,skip_special_tokens=True)
    
    score = corpus_bleu(outputs,[labels]).score
    return [score]


def main(args):
    alignment_strategy = args.alignment_strategy
    data_path = os.path.join(args.input_data_dir, args.data_file)
    target_dir = os.path.join(args.output_data_dir, f"{os.path.splitext(args.data_file)[0]}_{alignment_strategy}.jsonl")
    data = process_jsonl(data_path)
    if alignment_strategy  == "hallucination":
        ref_data = data
    elif alignment_strategy == "language":
        ref_data = process_jsonl(args.reference_en_file)
    elif alignment_strategy == "preference":
        ref_data = data
    else:
        raise NotImplementedError
    
    rm_model_base = AutoModelForSeq2SeqLM.from_pretrained(args.reward_model_path,device_map='auto')
    rm_model_base = rm_model_base.eval()
    rm_tokenizer = AutoTokenizer.from_pretrained(args.reward_model_path)

    if os.path.exists(target_dir):
        exist_data = process_jsonl(target_dir)
        fw = open(target_dir, "a+",encoding="utf8")
    else:
        os.makedirs(os.path.dirname(target_dir), exist_ok=True)
        fw= open(target_dir, "a+",encoding="utf8")
        exist_data = None
        
    for i in tqdm(range(len(data))):
        if exist_data is not None and i < len(exist_data) and exist_data[i]['question_id'] == data[i]['question_id']:
            continue
        item = data[i]
        ref_item = ref_data[i]
        assert item['question_id'] == ref_item['question_id']
        lang = nllb_200_distilled_600M_language_dict[language_dict[args.language]['full_name']]
        if lang == "eng_Latn":
            continue
        item_reward_list=[]
        for it2 in item['translation']: 
            it = it2["result"]
            reward_list = []
            en_answers = ref_item['answer'] if alignment_strategy == "language" else ref_item['en_answer']
            en_answers = en_answers if isinstance(en_answers, list) else [en_answers]
            input_answer = [it for _ in en_answers]
            output = [j for j in en_answers]
            if alignment_strategy == "hallucination":
                reject_en_answers = ref_item["reject_en_answer"]
                reject_en_answers = reject_en_answers if isinstance(en_answers,list) else [reject_en_answers]
                reject_input_answer = [it for _ in reject_en_answers]
                reject_output = [j for j in reject_en_answers]
                if len(reject_en_answers)<=20:
                    reject_reward_list = BLEU_Alighment_reward_fuction(rm_tokenizer,rm_model_base,reject_input_answer,reject_output,lang)
                else:
                    raise NotImplementedError
                reject_res = {
                    'reject-nllb-200-distilled-600M-reward-mean':sum(reject_reward_list)/len(reject_reward_list),
                    'reject-nllb-200-distilled-600M-reward-max':max(reject_reward_list),
                    'reject-nllb-200-distilled-600M-reawrdlist':reject_reward_list,
                }
            else:
                reject_res = {}
                
            if len(input_answer) <= 20:
                reward_list = BLEU_Alighment_reward_fuction(rm_tokenizer,rm_model_base,input_answer,output,lang)
            else:
                reward_list1 = BLEU_Alighment_reward_fuction(rm_tokenizer,rm_model_base,input_answer[:20],output[:20],lang)
                reward_list2 = BLEU_Alighment_reward_fuction(rm_tokenizer,rm_model_base,input_answer[20:],output[20:],lang)
                reward_list = reward_list1 + reward_list2
                assert len(reward_list) == len(output)
            reward_res = {
                'nllb-200-distilled-600M-reward-mean':sum(reward_list)/len(reward_list),
                'nllb-200-distilled-600M-reward-max':max(reward_list),
                'nllb-200-distilled-600M-reawrdlist':reward_list,
            }
            reward_res.update(reject_res)
            item_reward_list.append(reward_res)
            torch.cuda.empty_cache()
        item["reward_list"] = item_reward_list
        fw.write(json.dumps(item, ensure_ascii=False) + '\n')
    fw.close()

            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="sampling argument")
    parser.add_argument('-rm','--reward_model_path', type=str,default='/mnt/petrelfs/songmingyang/songmingyang/model/others/nllb-200-distilled-600M')
    parser.add_argument('-rf','--reference_en_file',type=str, default='/mnt/petrelfs/songmingyang/songmingyang/runs/llava/test/generations/llava_7b_v1_generation_num20_en.json')
    parser.add_argument('-i','--input_data_dir',type=str, default="/mnt/petrelfs/songmingyang/songmingyang/runs/llava/test/generations")
    parser.add_argument('-o','--output_data_dir',type=str, default="/mnt/petrelfs/songmingyang/songmingyang/runs/llava/test/add_ppl/20_en_refs/")
    parser.add_argument('-d','--data_file', type=str)
    parser.add_argument('-l','--language', type=str, default="en")
    parser.add_argument('-a','--alignment_strategy',type=str, default="hallucination")
    args = parser.parse_args()
    main(args)
            



