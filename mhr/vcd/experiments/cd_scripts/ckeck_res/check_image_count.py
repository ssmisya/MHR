import os
import glob
import json
from mhr.utils.utils import process_jsonl
from tqdm import tqdm

def count_lines_in_file(file_path):
    with open(file_path, 'r') as file:
        return sum(1 for line in file)
    
def examine_image_counts(file_path):
    data = process_jsonl(file_path)
    for idx,line in enumerate(data):
        question = line['question']
        chosen = line["feedback"]["chosen"]
        reject = line['feedback']['reject']
        for sentence in chosen + reject:
            if '<image>' in sentence:
                print(f'Chosen: {sentence} in the {idx}th line of {os.path.basename(file_path)}')
                
    
    
    
def main(directory):
    # 获取目录下所有的jsonl文件
    jsonl_files = glob.glob(os.path.join(directory, '*.jsonl'))
    for jsonl_file in tqdm(jsonl_files):
        examine_image_counts(jsonl_file)

if __name__ == '__main__':
    directory = f'/mnt/petrelfs/songmingyang/songmingyang/runs/llava/preprocess/sft_on_palo_1000/desc_dpo_data'  # 替换为你的目录路径
    main(directory)