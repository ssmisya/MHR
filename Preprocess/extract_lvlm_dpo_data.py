import json
import re
import argparse
import os


from experiments.eval.language_dict import language_dict

def main(args):
    input_file=os.path.join(args.input_dir,args.file_name)
    data = []
    with open(input_file, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line)
            data.append(json_obj)

    dpo_data = []
    english_instruction2data = {}

    for i in data:
        lang =  language_dict[args.language]['full_name']
        if lang != "English":
            output_with_score = [{"answer":i['answer'][k],'score':i['reward_list'][k]} for k in range(len(i['answer']))] 
            sorted_output = [g for g in sorted(output_with_score,key=lambda x:x['score']["nllb-200-distilled-600M-reward-mean"],reverse=True)]
            temp = english_instruction2data.get(i['en_question'],[])
            for j in range(len(sorted_output)-1):

                # TODO:判断不对的答案 删去，猜想LVLM应该做幻觉判断，但缺乏检测机制(GPT-4)，暂时跳过
                # predict_answer = extract_last_num(sorted_output[j]['generated'])
                # if abs(label - predict_answer) > 1e-3:
                #     continue
                
                for l in range(j+1,len(sorted_output)):
                    sample = {}
                    sample['accept'] = sorted_output[j]['answer']
                    sample['reject'] = sorted_output[l]['answer']
                    sample['score-diff'] = sorted_output[j]['score']['nllb-200-distilled-600M-reward-mean']-sorted_output[l]['score']['nllb-200-distilled-600M-reward-mean']
                    if sorted_output[j]['nllb-200-distilled-600M-reward-mean'] != sorted_output[l]['nllb-200-distilled-600M-reward-mean'] and process(sorted_output[j]['generated']) != process(sorted_output[l]['generated']):
                        temp.append(sample)
            english_instruction2data[i['en_question']] = temp


    ratio = 10
    index = 0
    train_data = []
    dev_data = []
    for i in english_instruction2data:
        if index % ratio == 0:
            dev_data.extend(english_instruction2data[i])
        else:
            train_data.extend(english_instruction2data[i])
        index += 1
    
    print(len(train_data))
    print(len(dev_data))
    f = open("/mnt/data/shesj/Data/RL4CoTData/rm_data/{}-onlycorrect-train.json".format(target),'w')
    json.dump(train_data,f,indent=2,ensure_ascii=False)
    f = open("/mnt/data/shesj/Data/RL4CoTData/rm_data/{}-onlycorrect-dev.json".format(target),'w')
    json.dump(dev_data,f,indent=2,ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    # parser.add_argument('integers', metavar='N', type=int, nargs='+',
    #                     help='an integer for the accumulator')
    # parser.add_argument('--sum', dest='accumulate', action='store_const',
    #                     const=sum, default=max,
    #                     help='sum the integers (default: find the max)')
    parser.add_argument('-l','--language', type=str, default="en")
    parser.add_argument('-i','--input_dir', type=str, default="/mnt/petrelfs/songmingyang/songmingyang/runs/llava/test/add_ppl")
    parser.add_argument('-o','--output_dir', type=str, default="/mnt/petrelfs/songmingyang/songmingyang/runs/llava/test/dpo_data")
    parser.add_argument('-n','--file_name',type=str,default="llava_7b_v1_generation_num20_bn.json_0_2000.jsonl")
    args = parser.parse_args()
    main(args)
