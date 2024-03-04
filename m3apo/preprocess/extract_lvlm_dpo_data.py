import json
import re
import argparse
import os
from tqdm import tqdm


from m3apo.vcd.experiments.eval.language_dict import language_dict
from m3apo.utils.utils import write_jsonl,process_jsonl

from langdetect import detect

def main(args):
    input_file=os.path.join(args.input_dir,args.file_name)
    data = process_jsonl(input_file)

    dpo_data = []
    for i in tqdm(data):
        feedback = []
        lang =  language_dict[args.language]['full_name']
        if lang != "English":
            output_with_score = [{"answer":i['answer'][k],'score':i['reward_list'][k]} for k in range(len(i['answer']))] 
            sorted_output = [g for g in sorted(output_with_score,key=lambda x:x['score']["nllb-200-distilled-600M-reward-mean"],reverse=True)]
            # temp = english_instruction2data.get(i['en_question'],[])
            
            for j in range(len(sorted_output)-1):
                # TODO:判断不对的答案 删去，猜想LVLM应该做幻觉判断，但缺乏检测机制(GPT-4)，暂时跳过
                # predict_answer = extract_last_num(sorted_output[j]['generated'])
                # if abs(label - predict_answer) > 1e-3:
                #     continue
                for l in range(j+1,len(sorted_output)):
                    sample = {}
                    sample['chosen'] = sorted_output[j]['answer']
                    sample['reject'] = sorted_output[l]['answer']
                    sample['score-diff'] = sorted_output[j]['score']['nllb-200-distilled-600M-reward-mean']-sorted_output[l]['score']['nllb-200-distilled-600M-reward-mean']
                    if sorted_output[j]['score']['nllb-200-distilled-600M-reward-mean'] != sorted_output[l]['score']['nllb-200-distilled-600M-reward-mean']:
                        feedback.append(sample)
        prompt_suffix = f" Please answer this question in {language_dict[args.language]['full_name']}"
        format_data = {
            "question":i['prompt']+prompt_suffix,
            "feedback":feedback,
            "image":i['image']
        }
        dpo_data.append(format_data)
    write_jsonl(dpo_data,os.path.join(args.output_dir,args.file_name))

def extract_top3(args):
    input_file=os.path.join(args.input_dir,args.file_name)
    data = process_jsonl(input_file)

    dpo_data = []
    for i in tqdm(data):
        output_with_score = [{"answer":i['answer'][k],'score':i['reward_list'][k]} for k in range(len(i['answer']))] 
        filtered_output_with_score = []
        for obj in output_with_score:
            try:
                sign = (detect(obj["answer"]) == args.language)
            except:
                sign = False
            if sign:
                filtered_output_with_score.append(obj)
        output_with_score = filtered_output_with_score
        if len(output_with_score) < 2:
            print(f"Warning: question_id:{i['question_id']} has less than 2 {args.language} answers")
            continue
            
        sorted_output = [g for g in sorted(output_with_score,key=lambda x:x['score']["nllb-200-distilled-600M-reward-mean"],reverse=True)]
        
        topk = min(3,len(sorted_output)//2)
        # test language and choose top 3

        reject_list=[sorted_output[-k_rej-1]['answer'] for k_rej in range(topk)]
        accept_list=[sorted_output[k]["answer"] for k in range(topk)]
        if len(reject_list) == 0 or len(accept_list) == 0:
            print(f"Warning: question_id:{i['question_id']} has less than 2 {args.language} answers")
            continue
        sample = {
                "chosen":accept_list,
                "reject":reject_list,
        }
        # construct data
        prompt_suffix = f" Please answer this question in {language_dict[args.language]['full_name']}"
        format_data = {
                "question_id":i['question_id'],
                "question":i['prompt']+prompt_suffix,
                "feedback":sample,
                "image":i['image'],
                "language":args.language,
        }
        dpo_data.append(format_data)
    write_jsonl(dpo_data,os.path.join(args.output_dir,args.file_name))

def extract_self_hallucinagion_top3(args):
    input_file=os.path.join(args.input_dir,args.file_name)
    data = process_jsonl(input_file)

    dpo_data = []
    for i in tqdm(data):
        output_with_score = [{"answer":i['answer'][k],'score':i['reward_list'][k]} for k in range(len(i['answer']))] 
        filtered_output_with_score = []
        for obj in output_with_score:
            try:
                sign = (detect(obj["answer"]) == args.language)
            except:
                sign = False
            if sign:
                filtered_output_with_score.append(obj)
        output_with_score = filtered_output_with_score
        if len(output_with_score) < 2:
            print(f"Warning: question_id:{i['question_id']} has less than 2 {args.language} answers")
            continue
            
        sorted_output = [g for g in sorted(output_with_score,key=lambda x:x['score']["nllb-200-distilled-600M-reward-mean"],reverse=True)]
        sorted_output_reject = [g for g in sorted(output_with_score,key=lambda x:x['score']["reject-nllb-200-distilled-600M-reward-mean"],reverse=True)]
        
        topk = min(3,len(sorted_output)//2)
        # test language and choose top 3

        reject_list=[sorted_output_reject[k_rej]['answer'] for k_rej in range(topk)]

        accept_list =[]
        k=0
        while len(accept_list) < topk and k < len(sorted_output):
            if sorted_output[k]['answer'] not in reject_list:
                accept_list.append(sorted_output[k]['answer'])
            k += 1
                
        if len(reject_list) == 0 or len(accept_list) == 0:
            print(f"Warning: question_id:{i['question_id']} has less than 2 {args.language} answers")
            continue
        sample = {
                "chosen":accept_list,
                "reject":reject_list,
        }
        # construct data
        prompt_suffix = f" Please answer this question in {language_dict[args.language]['full_name']}"
        format_data = {
                "question_id":i['question_id'],
                "question":i['prompt']+prompt_suffix,
                "feedback":sample,
                "image":i['image'],
                "language":args.language,
        }
        dpo_data.append(format_data)
    write_jsonl(dpo_data,os.path.join(args.output_dir,args.file_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-l','--language', type=str, default="en")
    parser.add_argument('-i','--input_dir', type=str, default="/mnt/petrelfs/songmingyang/songmingyang/runs/llava/test/add_ppl")
    parser.add_argument('-o','--output_dir', type=str, default="/mnt/petrelfs/songmingyang/songmingyang/runs/llava/test/dpo_data")
    parser.add_argument('-n','--file_name',type=str,default="llava_7b_v1_generation_num20_bn.json_0_2000.jsonl")
    parser.add_argument('-m',"--extract_method",type=str,default="extract_top3")
    # parser.add_argument('-r','--refrence_file',type=str,default="/mnt/petrelfs/songmingyang/songmingyang/runs/llava/test/generations/llava_7b_v1_generation_num20_en.json")
    args = parser.parse_args()
    if args.extract_method == "extract_top3":
        extract_top3(args)
    elif args.extract_method == "compare_with_everyone":
        main(args)
    elif args.extract_method == "extract_self_hallucinagion_top3":
        extract_self_hallucinagion_top3(args)
    else:
        raise ValueError("extract_method should be one of ['extract_top3','compare_with_everyone']")
