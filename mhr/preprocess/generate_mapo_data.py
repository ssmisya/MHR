import argparse
import random

import os
import json
from tqdm import tqdm
import shortuuid
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from mhr.vcd.experiments.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from mhr.vcd.experiments.llava.conversation import conv_templates, SeparatorStyle
from mhr.vcd.experiments.llava.model.builder import load_pretrained_model
from mhr.vcd.experiments.llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from mhr.vcd.experiments.llava.utils import disable_torch_init
from mhr.vcd.experiments.llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from mhr.vcd.vcd_utils.vcd_add_noise import add_diffusion_noise
from mhr.vcd.vcd_utils.vcd_sample import evolve_vcd_sampling
evolve_vcd_sampling()

from mhr.vcd.experiments.eval.language_dict import language_dict, nllb_200_distilled_600M_language_dict
from mhr.vcd.experiments.llava.model.language_model.llava_llama import  LlavaLlamaForCausalLM
from mhr.utils.utils import process_jsonl,load_json_file,write_jsonl
from mhr.utils.debugging import remote_breakpoint
from typing import Dict, Optional, List, Sequence

import torch
from torch.utils.data import DataLoader
import transformers
from transformers import PreTrainedModel, PreTrainedTokenizer, AutoTokenizer, AutoModelForCausalLM, AutoProcessor
from torch.utils.data import Dataset
from dataclasses import dataclass, field
from accelerate import Accelerator
from accelerate.utils import gather_object


from PIL import Image
import debugpy


def convert_dict_to_tensor(results, device):
    part_tensor = json.dumps(results)
    part_tensor = torch.Tensor([ord(part_tensor[i]) for i in range(len(part_tensor))]).long().to(device)
    return part_tensor

@dataclass
class ScriptArguments:
    middle_output_file_path: str = field(default="/mnt/petrelfs/songmingyang/songmingyang/runs/llava/dpo_data_process/human_preference/middle", metadata={"help": "Path to the middle output file."})

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="/mnt/petrelfs/songmingyang/songmingyang/model/others/llava-v1.5-7b")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default="/mnt/petrelfs/songmingyang/songmingyang/model/others/clip-vit-large-patch14-336")
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="patch")   

@dataclass
class DataArguments:
    
    data_path: str = field(default="/mnt/petrelfs/songmingyang/songmingyang/data/mm/annotation/LLaVA-Human-Preference-10K/llava_7b_v1_preference.json", metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default="/mnt/petrelfs/songmingyang/songmingyang/data/mm/imgs/train2017")
    image_aspect_ratio: str = 'square'
    language: str = 'bn'
    data_type: str = 'bf16'
    batch_size: int = 4
    num_workers: int = 0
    ref_data_path :Optional[str] = field(default="/mnt/petrelfs/songmingyang/songmingyang/runs/llava/test/generations/llava_7b_v1_generation_num20_en.json")

@dataclass
class DataCollatorForReferenceLanguageDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids= [instance["input_ids"] for instance in instances]
        
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        
        input_ids = input_ids[:, :self.tokenizer.model_max_length]

        batch = dict(
            input_ids=input_ids,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        return batch

@dataclass
class DataCollatorForAlignmentLanguageDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids= [instance["input_ids"] for instance in instances]
        
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        
        input_ids = input_ids[:, :self.tokenizer.model_max_length]

        question_ids = [i['question_id'] for i in instances]
        image_file_names = [i['image_file_name'] for i in instances]
        questions = [i['question'] for i in instances]
        preferred_en_answers = [i['preferred_en_answer'] for i in instances]
        hallucinations = [i['hallucination'] for i in instances]
        

        batch = dict(
            input_ids=input_ids,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            question_id=question_ids,
            image_file_name=image_file_names,
            question = questions,
            preferred_en_answer=preferred_en_answers,
            hallucination=hallucinations
        )

        if 'images' in instances[0]:
            images = [instance['images'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images
        
        return batch

class ReferenceLanguageDataset(Dataset):
    def __init__(self, 
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
        seed: int = 42,
    ):
        super(ReferenceLanguageDataset, self).__init__()
        random.seed(seed)
        self.tokenizer = tokenizer

        self.data_args = data_args
        list_data_dict = self.data_preprocess(data_path)
        self.list_data_dict = list_data_dict
        
    
    def generate_prompt(self,qs,language,mm_use_im_start_end=False,conv_mode="llava_v1"):
        if not DEFAULT_IMAGE_TOKEN in qs:
            if mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[conv_mode].copy()
        prompt_suffix = f" Please answer this question in {language_dict[language]['full_name']}"
        conv.append_message(conv.roles[0], qs + prompt_suffix)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        return prompt
    
    def data_preprocess(self,data_path):
        if os.path.basename(data_path) == "llava_7b_v1_preference.json":
            return self.preprocess_human_performance_10k(data_path)
        else:
            raise ValueError("The data file is not supported")
    
    def preprocess_human_performance_10k(self,data_path):
        data_dict_list=[]
        origin_data = load_json_file(data_path)
        for line in origin_data:
            idx = line['id']
            image_file = line['image']
            question_dict = line['conversations'][-2]
            assert question_dict['from'] == 'human'
            qs = question_dict['value']
            preferred_en_answer=line[f"output_{line['preference']}"]['value']
            prompt = self.generate_prompt(qs,self.data_args.language)
            data_dict_list.append({
                "question_id":idx,
                "image":image_file,
                "question":prompt,
                "preferred_en_answer":[preferred_en_answer],
                "hallucination":line['hallucination']
            })
        return data_dict_list
        
        
    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'images' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        input_ids= self.tokenizer(self.list_data_dict[i]["preferred_en_answer"], return_tensors='pt', padding='longest', truncation=True,max_length=512)
        if isinstance(i, int):
            data_dict = dict(
                input_ids=input_ids,
            )
        return data_dict


class AlignmentLanguageDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, 
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
        seed: int = 42,
    ):
        super(AlignmentLanguageDataset, self).__init__()
        random.seed(seed)
        self.tokenizer = tokenizer

        self.data_args = data_args
        list_data_dict = self.data_preprocess(data_path)
        self.list_data_dict = list_data_dict
        self.float_type = torch.bfloat16 if self.data_args.data_type=='bf16' else torch.float32
        
    
    def generate_prompt(self,qs,language,mm_use_im_start_end=False,conv_mode="llava_v1"):
        if not DEFAULT_IMAGE_TOKEN in qs:
            if mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[conv_mode].copy()
        prompt_suffix = f" Please answer this question in {language_dict[language]['full_name']}"
        conv.append_message(conv.roles[0], qs + prompt_suffix)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        return prompt
    
    def data_preprocess(self,data_path):
        if os.path.basename(data_path) == "llava_7b_v1_preference.json":
            return self.preprocess_human_performance_10k(data_path)
        else:
            raise ValueError("The data file is not supported")
    
    def preprocess_human_performance_10k(self,data_path):
        data_dict_list=[]
        origin_data = load_json_file(data_path)
        for line in origin_data:
            idx = line['id']
            image_file = line['image']
            question_dict = line['conversations'][-2]
            assert question_dict['from'] == 'human'
            qs = question_dict['value']
            preferred_en_answer=line[f"output_{line['preference']}"]['value']
            prompt = self.generate_prompt(qs,self.data_args.language)
            data_dict_list.append({
                "question_id":idx,
                "image":image_file,
                "question":prompt,
                "preferred_en_answer":preferred_en_answer,
                "hallucination":line['hallucination']
            })
        return data_dict_list
        
        
    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'images' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            else:
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            
        input_ids = tokenizer_image_token(self.list_data_dict[i]['question'], self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        
        if isinstance(i, int):
            data_dict = dict(
                question_id=self.list_data_dict[i]['question_id'],
                input_ids=input_ids,
                image_file_name=self.list_data_dict[i]['image'],
                question=self.list_data_dict[i]['question'],
                preferred_en_answer=self.list_data_dict[i]['preferred_en_answer'],
                hallucination=self.list_data_dict[i]['hallucination'],
            )

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['images'] = image.to(self.float_type)
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['images'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        return data_dict


def calculate_PPL_score(
    model : PreTrainedModel,
    tokenizer : PreTrainedTokenizer,
    ref_dataset : ReferenceLanguageDataset,
    alignment_dataset : AlignmentLanguageDataset,
    source_language : str = 'fr',
):
    if isinstance(model,transformers.M2M100ForConditionalGeneration):
        supplement_language_dict = nllb_200_distilled_600M_language_dict
    else:
        raise ValueError("The model is not supported")
    source_language = supplement_language_dict[language_dict[source_language]['full_name']]
    target_lang = 'eng_Latn'
   
    assert len(ref_dataset) == len(alignment_dataset)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    for i in range(len(ref_dataset)):
        outputs = ref_dataset[i]['input_ids']
        labels = ref_dataset[i]['input_ids']
        tokenizer.src_lang = source_language
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

def extract_dpo_data():
    pass
def process_image(processor,image_file_name,image_folder="/mnt/petrelfs/songmingyang/songmingyang/data/mm/imgs/train2017",dtype=torch.bfloat16):
    image = Image.open(os.path.join(image_folder, image_file_name))
    image_tensor = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
    return image_tensor.to(dtype)

def generate_prompt(qs,language,mm_use_im_start_end=False,conv_mode="llava_v1"):
        if not DEFAULT_IMAGE_TOKEN in qs:
            if mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[conv_mode].copy()
        prompt_suffix = f" Please answer this question in {language_dict[language]['full_name']}"
        conv.append_message(conv.roles[0], qs + prompt_suffix)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        return prompt
    
def data_preprocess(data_path,language,tokenizer,dtype=torch.bfloat16):
        if os.path.basename(data_path) == "llava_7b_v1_preference.json":
            return preprocess_human_performance_10k(data_path,language,tokenizer)
        else:
            raise ValueError("The data file is not supported")
    
def preprocess_human_performance_10k(data_path,language,tokenizer,dtype=torch.bfloat16):
        data_dict_list=[]
        origin_data = load_json_file(data_path)
        for line in origin_data:
            idx = line['id']
            image_file = line['image']
            question_dict = line['conversations'][-2]
            assert question_dict['from'] == 'human'
            qs = question_dict['value']
            preferred_en_answer=line[f"output_{line['preference']}"]['value']
            prompt = generate_prompt(qs,language)
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
            data_dict_list.append({
                "question_id":idx,
                "image":image_file,
                "question":prompt,
                "preferred_en_answer":preferred_en_answer,
                "hallucination":line['hallucination'],
                "input_ids":input_ids,
            })
        return data_dict_list

def convert_json_to_list(data):
    pass
    

def model_generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset,
    accelerator: Accelerator,
    image_processor,
    top_p: float = 1.0,
    top_k: int = None,
    temperature: float = 1.0,
    generation_num: int = 20,
    language: str = 'en',
    image_folder: str = "/mnt/petrelfs/songmingyang/songmingyang/data/mm/imgs/train2017",
):


    if torch.distributed.is_available() and torch.distributed.is_initialized():
        world_size=torch.distributed.get_world_size()
        current_process=torch.distributed.get_rank()
        trunk_size=len(dataset) // world_size
        start = current_process * trunk_size
        end = start + trunk_size if current_process < world_size - 1 else len(dataset)
    else :
        current_process="none"
        start = 0
        end = len(dataset)
    
    results=[]
    for i in tqdm(range(start,end),desc=f"Current Process: {current_process}"):
        data = dataset[i]
        input_ids = data['input_ids'].unsqueeze(0).to(accelerator.device)
        image_tensor = process_image(image_processor,data['image'],image_folder).unsqueeze(0).to(accelerator.device)
        
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                images_cd=None,
                cd_alpha = None,
                cd_beta =  None,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_new_tokens=512,
                use_cache=True,
                num_return_sequences=generation_num,  # 控制生成的序列数量
                )
            
            
        
        input_token_len = input_ids.shape[1] # input_ids: [batch_size, seq_len]
        # output_ids: [batch_size, seq_num, seq_len]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)
        prepared_output = {
            "outputs": outputs,
            "question_id":data['question_id'],
            "image":data['image'],
            "question":data['question'],
            "preferred_en_answer":data['preferred_en_answer'],
            "hallucination":data['hallucination'],
            "language":language,
        }
        results.append(prepared_output)
        
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        device = f"cuda:{torch.cuda.current_device()}"
        # convert dictionary -> tensor for gather all results in all ranks
        part_tensor = convert_dict_to_tensor(results, device)
        shape_tensor = torch.tensor(part_tensor.shape, device=device)
        shape_list = [shape_tensor.clone() for _ in range(int(os.environ["WORLD_SIZE"]))]
        torch.distributed.all_gather(shape_list, shape_tensor)

        # gather tensor
        max_shape = max(shape_list)
        part_tensor_pad = torch.zeros(max_shape).to(device)
        part_tensor_pad[:part_tensor.shape[0]] = part_tensor
        tensor_list = [part_tensor_pad.clone() for _ in range(int(os.environ["WORLD_SIZE"]))]
        torch.distributed.all_gather(tensor_list, part_tensor_pad)

        if int(os.environ["RANK"]) == 0:
            results_all_rank = []
            for tensor, shape in zip(tensor_list, shape_list):
                t = tensor.long()[:shape]
                _data = "".join([chr(t[i].item()) for i in range(t.shape[0])])
                _data = json.loads(_data)
                results_all_rank.extend(_data)
            # sort according to question_id
            results_all_rank = sorted(results_all_rank, key=lambda x:x["question_id"])
    else:
        results_all_rank = results

    return results_all_rank



def initialize_model(model_name_or_path):
    accelerator = Accelerator()
    model_path = os.path.expanduser(model_name_or_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name,device=accelerator.device)
    
    return tokenizer,model,image_processor,accelerator

def main():
    torch.cuda.empty_cache()
    # initialize model and data
    parser = transformers.HfArgumentParser(
        (ScriptArguments, ModelArguments, DataArguments))
    script_args, model_args, data_args = parser.parse_args_into_dataclasses()
    
    tokenizer,model,image_processor,accelerator = initialize_model(model_args.model_name_or_path)
    # data_args.image_processor = image_processor
    # dataset = AlignmentLanguageDataset(data_args.data_path,tokenizer,data_args)
    dataset = data_preprocess(data_args.data_path,data_args.language,tokenizer)
    
    # write middle output file
    res = model_generate(model,tokenizer,dataset,accelerator,image_processor=image_processor, language=data_args.language, image_folder=data_args.image_folder)
    output_file_name=os.path.join(script_args.middle_output_file_path, f"{os.path.splitext(os.path.basename(data_args.data_path))[0]}.jsonl")
    write_jsonl(res, output_file_name)
    


if __name__ == "__main__":
    main()
