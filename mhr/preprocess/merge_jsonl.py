from mhr.utils.utils import merge_jsonl

# 要合并的jsonl文件路径列表
filepaths ="/mnt/petrelfs/songmingyang/songmingyang/runs/llava/test/dpo_data"

# 合并后的输出文件路径
output_filepath = '/mnt/petrelfs/songmingyang/songmingyang/runs/llava/test/merged/llava_multilingual_dpo_data.jsonl'

# 调用函数进行合并
merge_jsonl(filepaths, output_filepath)
