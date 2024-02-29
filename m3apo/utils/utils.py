import json

def process_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # 解析每一行为一个JSON对象
            json_obj = json.loads(line)
            # 进行你需要的处理，这里我们只是将其添加到数据列表中
            data.append(json_obj)
    return data