import json
import os

def process_jsonl(file_path):
    '''
        将jsonl文件转换为装有dict的列表
    '''
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line)
            data.append(json_obj)
    return data

def write_jsonl(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        for item in data:
            line = json.dumps(item,ensure_ascii=False)
            file.write(line + '\n')

def merge_jsonl(input_file_dir, output_filepath):
    '''
        将一个文件夹内的所有jsonl文件合并为一个jsonl文件
    '''
    filepaths = [os.path.join(input_file_dir, file) for file in os.listdir(input_file_dir)]
    merged_data = []
    for filepath in filepaths:
        with open(filepath, 'r') as file:
            for line in file:
                data = json.loads(line)
                merged_data.append(data)
    
    with open(output_filepath, 'w') as output_file:
        for data in merged_data:
            output_file.write(json.dumps(data) + '\n')
