import argparse
import json
import transformers
import torch
import os

from dataclasses import dataclass, field
from typing import Any,Sequence,Dict
from tqdm import tqdm
from transformers import AutoTokenizer, M2M100ForConditionalGeneration,pipeline
from torch.utils.data import Dataset, DataLoader

from mhr.vcd.experiments.eval.language_dict import language_dict,nllb_200_distilled_600M_language_dict
from mhr.utils.utils import load_json_file,write_json_file,process_jsonl,write_jsonl
from langdetect import detect


class DescDataset(Dataset):
    def __init__(
            self,
            data_path="",
            data_type="desc",
            tokenizer=None,
            language=None,
        ) -> None:
        super().__init__()
        self.data_path = data_path
        self.data_type = data_type
        self.tokenizer = tokenizer
        self.language = language
        self.load_data(data_type,data_path)
        
    def load_data(self,data_type,data_path):
        if data_type == "desc":
            self.load_desc_data(data_path)
        elif data_type == "jsonl":
            self.load_jsonl_data(data_path)
        else:
            raise NotImplementedError("This data type is not implemented yet")
        
    def load_desc_data(self,data_path):
        data_map_list = []
        desc_data = load_json_file(data_path)
        for question_id,data in desc_data.items():
            for sentence in data["chosen"]:
                data_map_list.append({"question_id":question_id,"sentence":sentence,"type":"chosen"})
            for sentence in data["rejected"]:
                data_map_list.append({"question_id":question_id,"sentence":sentence,"type":"rejected"})
        self.data_map_list = data_map_list
        
    def load_jsonl_data(self,data_path):
        data_map_list = []
        desc_data = process_jsonl(data_path)
        lang_code = self.language if self.language != "zh" else "zh-cn"
        assert lang_code is not None
        for data in desc_data:
            question_id = data["question_id"]
            answers = data["answer"]
            for idx,answer in enumerate(answers):
                data_map_list.append({"question_id":question_id,"sentence":answer,"type":idx})
        self.data_map_list = data_map_list
    
    def __len__(self):
        return len(self.data_map_list)
    
    def __getitem__(self, index) -> Any:
        data = self.data_map_list[index]
        tokenized_sentence = self.tokenizer(data["sentence"], return_tensors='pt', padding='longest', truncation=True,max_length=512)
        data["tokenized_sentence"] = tokenized_sentence
        return data

@dataclass
class DataCollatorForDescDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids ,attention_masks = tuple([instance["tokenized_sentence"][key][0] for instance in instances]
            for key in ("input_ids", "attention_mask"))
        question_ids,types = tuple([instance[key] for instance in instances]
            for key in ("question_id","type"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        attention_masks = torch.nn.utils.rnn.pad_sequence(attention_masks,
                                                 batch_first=True,
                                                 padding_value=0)
        batch = dict(
            inputs=dict(
                input_ids = input_ids,
                attention_masks = attention_masks,
            ),
            question_ids = question_ids,
            types = types
        )
        return batch
    
def batch_translate(model,dataset,tokenizer,target_lang,batch_size=32,device="cuda"):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,collate_fn=DataCollatorForDescDataset(tokenizer))
    result_list = {
        "sentence":[],
        "type":[],
        "question_id":[],
    }
    for batch in tqdm(dataloader):
        inputs = batch["inputs"]["input_ids"].to(device)
        output = model.generate(inputs, forced_bos_token_id=tokenizer.lang_code_to_id[target_lang])
        output_text = tokenizer.batch_decode(output,skip_special_tokens=True)
        result_list["sentence"].extend(output_text)
        result_list["type"].extend(batch["types"])
        result_list["question_id"].extend(batch["question_ids"])
    return result_list

def reconstruct_data_dict(result_list):
    result_dataset = {}
    for question_id,sentence,type in zip(result_list["question_id"],result_list["sentence"],result_list["type"]):
        if question_id not in result_dataset:
            result_dataset[question_id] = {"chosen":[],"rejected":[]}
        if type == "chosen":
            result_dataset[question_id]["chosen"].append(sentence)
        elif type == "rejected":
            result_dataset[question_id]["rejected"].append(sentence)
    return result_dataset

def reconstruct_data_jsonl(result_list,ref_jsonl_file):
    jsonl_data = process_jsonl(ref_jsonl_file)
    result_idx_by_question_id = {}
    for question_id,result,idx in zip(result_list["question_id"],result_list["sentence"],result_list["type"]):
        if result_idx_by_question_id.get(question_id, None) is None:
            result_idx_by_question_id[question_id] = []
        result_idx_by_question_id[question_id].append({"result":result,"idx":idx})
    for k,item in result_idx_by_question_id.items():
        item.sort(key=lambda x:x["idx"])
        result_idx_by_question_id[k] = [x["result"] for x in item]
        
    for data in jsonl_data:
        res_list = result_idx_by_question_id.get(data["question_id"],[])
        data["translation"] = res_list
        
    return jsonl_data
    

def main(args):
    if os.path.exists(args.output_file_path):
        print("already translated")
        return
    source_lang = nllb_200_distilled_600M_language_dict[language_dict[args.source_language]['full_name']]
    target_lang = nllb_200_distilled_600M_language_dict[language_dict[args.target_language]['full_name']]
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = M2M100ForConditionalGeneration.from_pretrained(args.model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.source_lang = source_lang
    tokenizer.target_lang = target_lang
    # translator = pipeline('translation', model=model, tokenizer=tokenizer, src_lang=source_lang,tgt_lang=target_lang)
    if args.input_type == "desc":
        dataset = DescDataset(args.input_file_path,args.input_type,tokenizer)
        result_list = batch_translate(model,dataset,tokenizer,target_lang,args.batch_size,device)
        result_dataset=reconstruct_data_dict(result_list)
        write_json_file(result_dataset,args.output_file_path)
    elif args.input_type == "jsonl":
        dataset = DescDataset(args.input_file_path,args.input_type,tokenizer,language=args.source_language)
        result_list = batch_translate(model,dataset,tokenizer,target_lang,args.batch_size,device)
        result_dataset = reconstruct_data_jsonl(result_list,args.input_file_path)
        write_jsonl(result_dataset,args.output_file_path)
    else:
        raise NotImplementedError("This input type is not implemented yet")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m","--model_path", type=str, default="")
    parser.add_argument("-i","--input_file_path", type=str, default="input.txt")
    parser.add_argument("-o","--output_file_path", type=str, default="output.txt")
    parser.add_argument("-t","--input_type", type=str, default="desc")
    parser.add_argument("--source_language", type=str, default="en")
    parser.add_argument("--target_language", type=str, default="zh")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    main(args)
    