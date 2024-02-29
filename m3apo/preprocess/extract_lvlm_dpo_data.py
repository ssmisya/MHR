import json
import re
import argparse
import os
from tqdm import tqdm


from m3apo.vcd.experiments.eval.language_dict import language_dict
from m3apo.utils.utils import write_jsonl,process_jsonl

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-l','--language', type=str, default="en")
    parser.add_argument('-i','--input_dir', type=str, default="/mnt/petrelfs/songmingyang/songmingyang/runs/llava/test/add_ppl")
    parser.add_argument('-o','--output_dir', type=str, default="/mnt/petrelfs/songmingyang/songmingyang/runs/llava/test/dpo_data")
    parser.add_argument('-n','--file_name',type=str,default="llava_7b_v1_generation_num20_bn.json_0_2000.jsonl")
    args = parser.parse_args()
    main(args)
