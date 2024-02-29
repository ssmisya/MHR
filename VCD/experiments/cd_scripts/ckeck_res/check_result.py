import os
import glob

def count_lines_in_file(file_path):
    with open(file_path, 'r') as file:
        return sum(1 for line in file)

def main(directory):
    # 获取目录下所有的jsonl文件
    jsonl_files = glob.glob(os.path.join(directory, '*.jsonl'))
    for jsonl_file in jsonl_files:
        line_count = count_lines_in_file(jsonl_file)
        if line_count != 3000:
            print(f'The file {jsonl_file} has {line_count} lines.')
            # os.remove(jsonl_file)

if __name__ == '__main__':
    directory = '/mnt/petrelfs/songmingyang/code/VCD/experiments/output/cd/'  # 替换为你的目录路径
    main(directory)