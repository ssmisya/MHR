import os
os.environ["WANDB_PROJECT"]="m3apo"

import json
import copy
import random
import logging
import argparse
import numpy as np
from PIL import Image
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Sequence

import torch
from torch.utils.data import Dataset

import transformers
from transformers import TrainerCallback
from transformers import HfArgumentParser, TrainingArguments

# from m3apo.alignment.models.llava_v1_5.llava.model import *
# form m3apo.vcd.experiments.llava.model import *

from m3apo.alignment.models.llava_v1_5.llava.model import *
from m3apo.alignment.models.llava_v1_5.llava import conversation as conversation_lib
from m3apo.alignment.models.llava_v1_5.llava.train.train import preprocess_multimodal, preprocess
from m3apo.alignment.models.llava_v1_5.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from m3apo.vcd.experiments.eval.language_dict import language_dict

from peft.peft_model import PeftModelForCausalLM
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)

from m3apo.alignment.trainer.llava_dpo_trainer import LlavaDPOTrainer
from m3apo.utils.utils import process_jsonl,load_json_file
from m3apo.utils.debugging import remote_breakpoint

local_rank = None
        
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="patch") 

@dataclass
class DataArguments:
    
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    vg_path: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    
    
    hallucination_data_path: Optional[str] = field(default=None, metadata={"help": "Path to the hallucination data."})
    hallucination_data_type: Optional[str] = field(default="dir_of_json_desc", metadata={"help": "Type of hallucination data."})
    hallucination_ratio: Optional[int] = field(default=0, metadata={"help": "The ratio of hallucination data."})
    
    language_data_path: Optional[str] = field(default=None, metadata={"help": "Path to the language data."})
    language_data_type: Optional[str] = field(default="dir_of_json_desc", metadata={"help": "Type of language data."})
    language_ratio: Optional[int] = field(default=0, metadata={"help": "The ratio of language data."})
    
    preference_data_path: Optional[str] = field(default=None, metadata={"help": "Path to the preference data."})
    preference_data_type: Optional[str] = field(default="dir_of_json_desc", metadata={"help": "Type of preference data."})
    preference_ratio: Optional[int] = field(default=0, metadata={"help": "The ratio of preference data."})
    
    translation_data_path: Optional[str] = field(default=None, metadata={"help": "Path to the translation data."})
    translation_data_type: Optional[str] = field(default="dir_of_json_desc", metadata={"help": "Type of translation data."})
    translation_ratio: Optional[int] = field(default=0, metadata={"help": "The ratio of translation data."})
    

# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """
    
    
    resume_from_checkpoint: Optional[bool] = field(default=False, metadata={"help": "Whether to resume from checkpoint."})
    # llava parameters
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: Optional[bool] = field(default=False, metadata={"help": "whether using lora fine-tuning model."})
    lora_r: Optional[int] = field(default=64, metadata={"help": "lora rank."})
    lora_alpha: Optional[int] = field(default=16, metadata={"help": "lora alpha."})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "lora dropout."})
    lora_weight_path: Optional[str] = field(default=None, metadata={"help": "path to lora weight."})
    lora_bias: Optional[str] = field(default="none", metadata={"help": "lora bias."})
    mm_projector_lr: Optional[float] = field(default=None, metadata={"help": "mm_projector learning rate."})
    group_by_modality_length: Optional[bool] = field(default=False, metadata={"help": "group_by_modality_length."})
    
    # beta
    beta: Optional[float] = field(default=0.5, metadata={"help": "the beta parameter for DPO loss"})
    
    # training parameters
    learning_rate: Optional[float] = field(default=5e-4, metadata={"help": "optimizer learning rate"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
    warmup_steps: Optional[int] = field(default=100, metadata={"help": "the number of warmup steps"})
    weight_decay: Optional[float] = field(default=0.05, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default="adamw_torch", metadata={"help": "the optimizer type"})
    max_grad_norm: Optional[float] = field(default=1.0, metadata={"help": "maximum value of gradient norm"})
    per_device_train_batch_size: Optional[int] = field(default=4, metadata={"help": "train batch size per device"})
    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "eval batch size per device"})
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "the number of gradient accumulation steps"}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "whether to use gradient checkpointing"}
    )
    ddp_find_unused_parameters: Optional[bool] = field(
        default=True, metadata={"help": "whether to find unused parameters. set to False when `gradient_checkpointing` is False."}
    )
    max_prompt_length: Optional[int] = field(default=512, metadata={"help": "the maximum prompt length"})
    max_length: Optional[int] = field(default=1024, metadata={"help": "the maximum sequence length"})
    max_steps: Optional[int] = field(default=-1, metadata={"help": "max number of training steps"})
    logging_steps: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})
    save_steps: Optional[int] = field(default=-1, metadata={"help": "the saving frequency"})
    evaluation_strategy: Optional[str] = field(default='no', metadata={"help": "the evaluation strategy"})
    eval_steps: Optional[int] = field(default=-1, metadata={"help": "the evaluation frequency"})
    output_dir: Optional[str] = field(default="./results", metadata={"help": "the output directory"})
    log_freq: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})
    deepspeed: Optional[str] = field(default=None, metadata={"help": "path to deepspeed config"})
    bf16: Optional[bool] = field(default=False, metadata={"help": "whether to use bf16 weight"})
    fp16: Optional[bool] = field(default=False, metadata={"help": "whether to use fp16 weight"})
    num_train_epochs: Optional[int] = field(default=3, metadata={"help": "number of training epochs"})
    save_strategy: Optional[str] = field(default="steps", metadata={"help": "strategy used to save model"})
    save_total_limit: Optional[int] = field(default=1, metadata={"help": "limit number of saved model"})
    num_train_epochs: Optional[int] = field(default=3, metadata={"help": "number of training epochs"})
    warmup_ratio: Optional[float] = field(default=0.03, metadata={"help": "warmup ratio"})
    tf32: Optional[bool] = field(default=True, metadata={"help": "whether to use tf32"})
    dataloader_num_workers: Optional[int] = field(default=4, metadata={"help": "number of dataloader workers"})
    fsdp: Optional[str] = field(default='', metadata={"help": "whether to use fsdp"})
    local_rank: int = field(default=-1, metadata={"help": "local rank"})
    seed: Optional[int] = field(default=42, metadata={"help": "seed"})
    
    # instrumentation
    report_to: Optional[str] = field(
        default="wandb",
        metadata={
            "help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,'
            '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. '
            'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'
        },
    )
    run_name: Optional[str] = field(default="dpo_llava-1.5", metadata={"help": "name of the run"})
    
    # debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)                                      

class MultilingualEnhanceDataset(Dataset):
    def __init__(self,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
        seed: int = 42,
    ):
        super(MultilingualEnhanceDataset, self).__init__()                                
        random.seed(seed)
        self.data_args = data_args
        self.tokenizer = tokenizer
        self.rng = random.Random(seed)
        self.initialize_data_args()
        
    
    def sample(self,i):
        def map_to_new_range(x, b, d):
            return int(x * (d / b))
        idx = i%self.max_data_lenth
        sample_weights=[self.hallucination_ratio,self.language_ratio,self.preference_ratio,self.translation_ratio]
        candidata_dataset = [self.hallucination_data,self.language_data,self.preference_data,self.translation_data]
        choice_dataset = self.rng.choices(candidata_dataset,sample_weights, k=1)[0]
        choice_length = len(choice_dataset)
        assert choice_length > 0 , "No data loaded."
        return choice_dataset[map_to_new_range(idx,self.max_data_lenth,choice_length,)]
        
        
       
            
        
        
        
    def initialize_data_args(self):
        # data paths
        self.hallucination_data_path = self.data_args.hallucination_data_path
        self.language_data_path = self.data_args.language_data_path
        self.preference_data_path = self.data_args.preference_data_path
        self.translation_data_path = self.data_args.translation_data_path
        
        self.hallucination_ratio = self.data_args.hallucination_ratio if self.hallucination_data_path else 0
        self.language_ratio = self.data_args.language_ratio if self.language_data_path else 0
        self.preference_ratio = self.data_args.preference_ratio if self.preference_data_path else 0
        self.translation_ratio = self.data_args.translation_ratio if self.translation_data_path else 0
        
        self.hallucination_data_type = self.data_args.hallucination_data_type if self.hallucination_data_path else None
        self.language_data_type = self.data_args.language_data_type if self.language_data_path else None
        self.preference_data_type = self.data_args.preference_data_type if self.preference_data_path else None
        self.translation_data_type = self.data_args.translation_data_type if self.translation_data_path else None
        
        if self.hallucination_data_path:
            self.hallucination_data = self.load_data_from_disk(self.hallucination_data_path,self.hallucination_data_type)
        else:
            self.hallucination_data = []
        
        if self.language_data_path:
            self.language_data = self.load_data_from_disk(self.language_data_path,self.language_data_type)
        else:
            self.language_data = []
            
        if self.preference_data_path:
            self.preference_data = self.load_data_from_disk(self.preference_data_path,self.preference_data_type)
        else:
            self.preference_data = []
        
        if self.translation_data_path:
            self.translation_data = self.load_data_from_disk(self.translation_data_path,self.translation_data_type)
        else:
            self.translation_data = []
        self.max_data_lenth = max(len(self.hallucination_data),len(self.language_data),len(self.preference_data),len(self.translation_data))
        
        # self.list_data_dict = []
        # if self.hallucination_data:
        #     self.list_data_dict += self.hallucination_data * 1
        # if self.language_data:
        #     self.list_data_dict += self.language_data * 1
        # if self.preference_data:
        #     self.list_data_dict += self.preference_data * 1
        
        # assert len(self.list_data_dict) > 0, "No data loaded."
        assert self.max_data_lenth > 0, "No data loaded."
        
        
    
    def load_data_from_disk(self,data_path,data_type):
        if data_type == "dir_of_json_desc":
            return self.load_dir_of_json_desc_data(data_path)
        elif data_type == "dir_of_jsonl_desc":
            return self.load_dir_of_jsonl_desc_data(data_path)
        else:
            raise NotImplementedError("This data type is not implemented yet")
            
    def load_dir_of_json_desc_data(self,data_path):
        vg_image_data = json.load(open(os.path.join(self.data_args.vg_path, "image_data.json")))
        id2path = {
            _data["image_id"]:os.path.join(self.data_args.image_folder, _data["url"].split("/")[-2], _data["url"].split("/")[-1]) 
            for _data in vg_image_data
        }
        data_dir = [os.path.join(data_path,a) for a in os.listdir(data_path)]
        data_across_different_langs = [load_json_file(a) for a in data_dir]
        data_dict_list = []
        id = 0
        for idx,file in enumerate(data_across_different_langs):
            file_name = os.path.basename(data_dir[idx])
            file_name_without_extension, extension = os.path.splitext(file_name)
            language = file_name_without_extension.strip().split("_")[-1]
            language_based_suffix = f" Please answer this question in {language_dict[language]['full_name']}."
            for question_id,data in file.items():
                # image_file_path = f"/mnt/petrelfs/songmingyang/songmingyang/data/mm/imgs/vg/VG_100K/{question_id}.jpg"
                image_file_path = id2path[int(question_id)]
                chosen = data["chosen"]
                reject = data["rejected"]
                question = self.rng.choice([
                        "Describe this image in detail.",
                        "Take a look at this image and describe what you notice.",
                        "Please provide a detailed description of the picture.",
                        "Could you describe the contents of this image for me?",
                    ])
                question += language_based_suffix
                data_dict_list.append(self.formulate_data_item(id,image_file_path,question,chosen,reject))
                id+=1
        return data_dict_list
    
    def load_dir_of_jsonl_desc_data(self,data_path):
        data_dir = [os.path.join(data_path,a) for a in os.listdir(data_path)]
        data_across_different_langs = [process_jsonl(a) for a in data_dir]
        data_dict_list = []
        id=0
        for file in data_across_different_langs:
            for data in file:
                image_file_path = data["image"]
                chosen = data["feedback"]["chosen"]
                reject = data["feedback"]["reject"]
                question = data["question"]
                data_dict_list.append(self.formulate_data_item(id,image_file_path,question,chosen,reject))
                id+=1
        return data_dict_list
    
    def formulate_data_item(self,id,image_file_or_path,question,chosen,reject):
        if not isinstance(chosen,list):
            chosen = [chosen]
        if not isinstance(reject,list):
            reject = [reject]
        if not DEFAULT_IMAGE_TOKEN in question:
            question = DEFAULT_IMAGE_TOKEN + '\n' + question
        elif question.count(DEFAULT_IMAGE_TOKEN) > 1:
            question = question.replace(DEFAULT_IMAGE_TOKEN,"")
            question = question.replace("\n","")
            question = DEFAULT_IMAGE_TOKEN + '\n' + question
        self.remove_image_token(chosen)
        self.remove_image_token(reject)
            
        questions_and_feedbacks = {"question":question,"feedback":{"chosen":chosen,"reject":reject}}
        res = {
                "id": int(id),
                "image": image_file_or_path,
                # "chosen_conversations": [
                #             {"from": "human", "value": "{question}"},
                #             {"from": "gpt", "value": "{chosen}"},
                #         ],
                # "reject_conversations": [
                #             {"from": "human", "value": "{question}"},
                #             {"from": "gpt", "value": "{reject}"},
                #         ],
                "questions_and_feedbacks":questions_and_feedbacks,
            }
        return res
    

    def remove_image_token(self, sentence_list):
        for i in range(len(sentence_list)):
            assert isinstance(sentence_list[i],str)
            if DEFAULT_IMAGE_TOKEN in sentence_list[i]:
                sentence_list[i] = sentence_list[i].replace(DEFAULT_IMAGE_TOKEN, "")
    
    def __len__(self):
        return self.max_data_lenth*(self.hallucination_ratio + self.language_ratio + self.preference_ratio + self.translation_ratio)

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
        # sources = self.list_data_dict[i]
        sources = self.sample(i)
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        
        questions_and_feedbacks = sources[0].get("questions_and_feedbacks",None)
        if questions_and_feedbacks is not None:
            # language_set = list(questions_and_feedbacks.keys())
            # language = random.choice(language_set)
            # print("dataset length:",len(self.list_data_dict),)
            question = questions_and_feedbacks["question"]
            chosen = self.rng.choice(questions_and_feedbacks["feedback"]["chosen"])
            reject = self.rng.choice(questions_and_feedbacks["feedback"]["reject"])
            chosen_conversations = [{"from": "human", "value": question},{"from": "gpt", "value": chosen}]
            reject_conversations =  [{"from": "human", "value": question},{"from": "gpt", "value": reject}]
            if question.count(DEFAULT_IMAGE_TOKEN) != 1 or question.count("\n") !=1:
                print(f"error on question: {question}, image: {sources[0]['image']}")
            # print(chosen_conversations)
            chosen_sources = preprocess_multimodal(
                copy.deepcopy([chosen_conversations for _ in sources]),
                self.data_args)
            reject_sources = preprocess_multimodal(
                copy.deepcopy([reject_conversations for _ in sources]),
                self.data_args)
            
        else:
            chosen_sources = preprocess_multimodal(
                copy.deepcopy([e["chosen_conversations"] for e in sources]),
                self.data_args)
            reject_sources = preprocess_multimodal(
                copy.deepcopy([e["reject_conversations"] for e in sources]),
                self.data_args)
        # remote_breakpoint()
        if 'image' in sources[0]:
            image_file = sources[0]['image']
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            image_path = image_file if os.path.isabs(image_file) else os.path.join(image_folder, image_file)
            image = Image.open(image_path).convert('RGB')
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
            
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        chosen_data_dict = preprocess(
            chosen_sources,
            self.tokenizer,
            has_image=('image' in sources[0]))
        reject_data_dict = preprocess(
            reject_sources,
            self.tokenizer,
            has_image=('image' in sources[0]))
        if isinstance(i, int):
            data_dict = dict(
                chosen_input_ids=chosen_data_dict["input_ids"][0],
                chosen_labels=chosen_data_dict["labels"][0],
                reject_input_ids=reject_data_dict["input_ids"][0],
                reject_labels=reject_data_dict["labels"][0],
            )

        # image exist in the data
        
        if 'image' in sources[0]:
            data_dict['images'] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['images'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        
        return data_dict
                
                    
        
class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, 
        data_path: str,
        self_hallucination_data_path: str,
        human_prefer_data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
        seed: int = 42,
    ):
        super(LazySupervisedDataset, self).__init__()
        random.seed(seed)
        if human_prefer_data_path:
            list_data_dict = self.human_preference_data_process(human_prefer_data_path)*3
        elif self_hallucination_data_path:
            list_data_dict = self.self_hallucination_data_process_pile(self_hallucination_data_path)*3
        elif data_path:
            list_data_dict = self.data_process(data_path)
        else:
            raise ValueError("Please provide a valid data path.")
        
        # if human_prefer_data_path:
        #     list_data_dict = list_data_dict + self.human_preference_data_process(human_prefer_data_path)
        
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
    
    def self_hallucination_data_process_pile(self,data_path):
        data_dir = [os.path.join(data_path,a) for a in os.listdir(data_path)]
        
        data_across_different_langs = [process_jsonl(a) for a in data_dir]
        id = 0
        data_dict_list = []
        for file in data_across_different_langs:
            for idx,data in enumerate(file):
                image_file_path = data["image"]
                questions_and_feedbacks = {}
                question = data["question"]
                # if not DEFAULT_IMAGE_TOKEN in question:
                #     question = DEFAULT_IMAGE_TOKEN + '\n' + question
                questions_and_feedbacks[data["language"]] = {"question":question,"feedback":data["feedback"]}
                data_dict_list.append({
                        "id": int(id),
                        "image": image_file_path,
                        "chosen_conversations": [
                            {"from": "human", "value": "{question}"},
                            {"from": "gpt", "value": "{chosen}"},
                        ],
                        "reject_conversations": [
                            {"from": "human", "value": "{question}"},
                            {"from": "gpt", "value": "{reject}"},
                        ],
                        "questions_and_feedbacks":questions_and_feedbacks,
                })
                id+=1
        return data_dict_list
        
    def human_preference_data_process(self,data_path):
        data_dir = [os.path.join(data_path,a) for a in os.listdir(data_path)]
        
        data_across_different_langs = [process_jsonl(a) for a in data_dir]

        data_dict_list = []
        id = 0
        for language_data in data_across_different_langs:
            for data in language_data:
                question_id = data["question_id"]
                image_file_path = data["image"]
                questions_and_feedbacks = {}
                question = data["question"]
                if not DEFAULT_IMAGE_TOKEN in question:
                    question = DEFAULT_IMAGE_TOKEN + '\n' + question
                questions_and_feedbacks[data["language"]] = {"question":question,"feedback":data["feedback"]}
                    
                data_dict_list.append({
                        "id": int(id),
                        "image": image_file_path,
                        "chosen_conversations": [
                            {"from": "human", "value": "{question}"},
                            {"from": "gpt", "value": "{chosen}"},
                        ],
                        "reject_conversations": [
                            {"from": "human", "value": "{question}"},
                            {"from": "gpt", "value": "{reject}"},
                        ],
                        "questions_and_feedbacks":questions_and_feedbacks,
                })
                id+=1
        return data_dict_list
    
    def self_hallucination_data_process(self,data_path):
        data_dir = [os.path.join(data_path,a) for a in os.listdir(data_path)]
        
        data_across_different_langs = [process_jsonl(a) for a in data_dir]
        length_list = [len(data) for data in data_across_different_langs]
        ids2feedbacks = [{_data["question_id"]:{"language":_data["language"],"question":_data["question"],"feedback":_data["feedback"]} for _data in data} for data in data_across_different_langs]
        data = data_across_different_langs[length_list.index(max(length_list))]
        
        data_dict_list = []
        id = 0
        for idx in range(len(data)):
            question_id = data[idx]["question_id"]
            image_file_path = data[idx]["image"]
            questions_and_feedbacks = {}
            for _data in ids2feedbacks:
                item = _data.get(question_id,None)
                if item is not None:
                    question = item["question"]
                    # if not DEFAULT_IMAGE_TOKEN in question:
                    #     question = DEFAULT_IMAGE_TOKEN + '\n' + question
                    questions_and_feedbacks[item["language"]] = {"question":question,"feedback":item["feedback"]}
            data_dict_list.append({
                    "id": int(id),
                    "image": image_file_path,
                    "chosen_conversations": [
                        {"from": "human", "value": "{question}"},
                        {"from": "gpt", "value": "{chosen}"},
                    ],
                    "reject_conversations": [
                        {"from": "human", "value": "{question}"},
                        {"from": "gpt", "value": "{reject}"},
                    ],
                    "questions_and_feedbacks":questions_and_feedbacks,
            })
            id+=1
        return data_dict_list
        
        
    def data_process(self, data_path):
        '''
            data: 是一个list, 每个元素是一个dict, 其中dict至少包含以下信息:
            {
                "question":prompt of this question,
                "feedback":[
                    {
                    "chosen":chosen answer1,
                    "reject":reject answer1,
                    },
                    {
                    "chosen":chosen answer2,
                    "reject":reject answer2,
                    }
                ]
                "image":image file name
                ...
            }
        '''
        data = process_jsonl(data_path)
        data_dict_list = []
        id = 0
        chosen_sign="chosen" if "chosen" in data[0]['feedback'][0].keys() else "accept"
        for idx in range(len(data)):
            for inner_idx in range(len(data[idx]['feedback'])):
                image_id = data[idx]["image"]
                chosen = data[idx]['feedback'][inner_idx][chosen_sign]
                reject = data[idx]['feedback'][inner_idx]["reject"]
                question = data[idx]["question"]
                if not DEFAULT_IMAGE_TOKEN in question:
                    question = DEFAULT_IMAGE_TOKEN + '\n' + question
                data_dict_list.append({
                    "id": int(id),
                    "image": image_id,
                    "chosen_conversations": [
                        {"from": "human", "value": question},
                        {"from": "gpt", "value": chosen},
                    ],
                    "reject_conversations": [
                        {"from": "human", "value": question},
                        {"from": "gpt", "value": reject},
                    ],
                })
                id+=1
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
        
        questions_and_feedbacks = sources[0].get("questions_and_feedbacks",None)
        if questions_and_feedbacks is not None:
            language_set = list(questions_and_feedbacks.keys())
            language = random.choice(language_set)
            # print("dataset length:",len(self.list_data_dict),)
            question = questions_and_feedbacks[language]["question"]
            chosen = random.choice(questions_and_feedbacks[language]["feedback"]["chosen"])
            reject = random.choice(questions_and_feedbacks[language]["feedback"]["reject"])
            chosen_conversations = [{"from": "human", "value": question},{"from": "gpt", "value": chosen}]
            reject_conversations =  [{"from": "human", "value": question},{"from": "gpt", "value": reject}]
            if question.count(DEFAULT_IMAGE_TOKEN) != 1 or question.count("\n") !=1:
                print(f"error on question: {question}, image: {sources[0]['image']}")
            # print(chosen_conversations)
            chosen_sources = preprocess_multimodal(
                copy.deepcopy([chosen_conversations for _ in sources]),
                self.data_args)
            reject_sources = preprocess_multimodal(
                copy.deepcopy([reject_conversations for _ in sources]),
                self.data_args)
            
        else:
            chosen_sources = preprocess_multimodal(
                copy.deepcopy([e["chosen_conversations"] for e in sources]),
                self.data_args)
            reject_sources = preprocess_multimodal(
                copy.deepcopy([e["reject_conversations"] for e in sources]),
                self.data_args)
        # remote_breakpoint()
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            image_path = image_file if os.path.isabs(image_file) else os.path.join(image_folder, image_file)
            image = Image.open(image_path).convert('RGB')
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
            
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        chosen_data_dict = preprocess(
            chosen_sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]))
        reject_data_dict = preprocess(
            reject_sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]))
        if isinstance(i, int):
            data_dict = dict(
                chosen_input_ids=chosen_data_dict["input_ids"][0],
                chosen_labels=chosen_data_dict["labels"][0],
                reject_input_ids=reject_data_dict["input_ids"][0],
                reject_labels=reject_data_dict["labels"][0],
            )

        # image exist in the data
        
        if 'image' in self.list_data_dict[i]:
            data_dict['images'] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['images'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        chosen_input_ids, chosen_labels, reject_input_ids, reject_labels = tuple([instance[key] for instance in instances]
            for key in ("chosen_input_ids", "chosen_labels", "reject_input_ids", "reject_labels"))
        chosen_input_ids = torch.nn.utils.rnn.pad_sequence(
            chosen_input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        chosen_labels = torch.nn.utils.rnn.pad_sequence(chosen_labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        reject_input_ids = torch.nn.utils.rnn.pad_sequence(
            reject_input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        reject_labels = torch.nn.utils.rnn.pad_sequence(reject_labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        chosen_input_ids = chosen_input_ids[:, :self.tokenizer.model_max_length]
        chosen_labels = chosen_labels[:, :self.tokenizer.model_max_length]
        reject_input_ids = reject_input_ids[:, :self.tokenizer.model_max_length]
        reject_labels = reject_labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            chosen_input_ids=chosen_input_ids,
            chosen_labels=chosen_labels,
            reject_input_ids=reject_input_ids,
            reject_labels=reject_labels,
            chosen_attention_mask=chosen_input_ids.ne(self.tokenizer.pad_token_id),
            reject_attention_mask=reject_input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'images' in instances[0]:
            images = [instance['images'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    # train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
    #                             data_path=data_args.data_path,
    #                             human_prefer_data_path=data_args.human_prefer_data_path,
    #                             self_hallucination_data_path=data_args.self_hallucination_data_path,
    #                             data_args=data_args)
    train_dataset = MultilingualEnhanceDataset(tokenizer=tokenizer,data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return

def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return

class SaverCallback(TrainerCallback):
    
    "A callback that prints a message at the end of training"
    def on_train_end(self, args, state, control, **kwargs):
        # save model
        if isinstance(kwargs['model'], PeftModelForCausalLM):
            torch.cuda.synchronize()
            state_dict = get_peft_state_maybe_zero_3(
                kwargs['model'].named_parameters(), "none"
            )
            kwargs['model'].save_pretrained(args.output_dir)
            non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
                kwargs['model'].named_parameters()
            )
            kwargs['model'].config.save_pretrained(args.output_dir)
            kwargs['model'].save_pretrained(args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(args.output_dir, 'non_lora_trainables.bin'))
    
def setup_llava_model(model_args, data_args, script_args):
    # local rank
    if "LOCAL_RANK" not in os.environ:
        local_rank = None
    else:
        local_rank = int(os.environ["LOCAL_RANK"])
    
    # device
    if "LOCAL_RANK" not in os.environ:
        device = f"cuda:{torch.cuda.current_device()}"
    else:
        device = f"cuda:{local_rank}"
    
    compute_dtype = (torch.float16 if script_args.fp16 else (torch.bfloat16 if script_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if script_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": device},
            load_in_4bit=script_args.bits == 4,
            load_in_8bit=script_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=script_args.bits == 4,
                load_in_8bit=script_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=script_args.double_quant,
                bnb_4bit_quant_type=script_args.quant_type # {'fp4', 'nf4'}
            )
        ))

    if model_args.vision_tower is not None:
        if 'mpt' in model_args.model_name_or_path:
            config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
            config.attn_config['attn_impl'] = script_args.mpt_attn_impl
            model = LlavaMPTForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=script_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
        else:
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=script_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=script_args.cache_dir,
            **bnb_model_from_pretrained_args
        )
    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if script_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if script_args.fp16 else (torch.bfloat16 if script_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=script_args.gradient_checkpointing)

    if script_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if script_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=script_args.lora_r,
            lora_alpha=script_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=script_args.lora_dropout,
            bias=script_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if script_args.bits == 16:
            if script_args.bf16:
                model.to(torch.bfloat16)
            if script_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    if 'mpt' in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=script_args.cache_dir,
            model_max_length=script_args.model_max_length,
            padding_side="right"
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=script_args.cache_dir,
            model_max_length=script_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=script_args.fsdp
        )
        
        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if script_args.bf16 else torch.float16, device=device)

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        model.config.tune_mm_mlp_adapter = script_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = script_args.freeze_mm_mlp_adapter
        if script_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        if script_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = script_args.mm_projector_lr
        script_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    if script_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if script_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if script_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)
                        
    return model, tokenizer
    
    
def main():
    # allow auto-dl completes on main process without timeout when using NCCL backend.
    # os.environ["NCCL_BLOCKING_WAIT"] = "1"
    parser = transformers.HfArgumentParser(
        (ScriptArguments, ModelArguments, DataArguments))
    script_args, model_args, data_args = parser.parse_args_into_dataclasses()
    
    # setup llava model
    llava_policy_model, tokenizer = setup_llava_model(
        model_args=model_args, 
        data_args=data_args,
        script_args=script_args,
    )
    script_args.lora_enable = False
    llava_ref_model, _ = setup_llava_model(
        model_args=model_args, 
        data_args=data_args,
        script_args=script_args,
    )
    
    # freeze reference model
    for n,p in llava_ref_model.named_parameters():
        p.requires_grad = False
    
    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)
    
    # if not use gradient_checkpointing, do not set ddp_find_unused_parameters
    if not script_args.gradient_checkpointing:
        script_args.ddp_find_unused_parameters = False
    
    # initialize training arguments:
    training_args = TrainingArguments(
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        max_steps=script_args.max_steps,
        logging_steps=script_args.logging_steps,
        save_steps=script_args.save_steps,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        ddp_find_unused_parameters=script_args.ddp_find_unused_parameters,
        learning_rate=script_args.learning_rate,
        evaluation_strategy=script_args.evaluation_strategy,
        eval_steps=script_args.eval_steps,
        output_dir=script_args.output_dir,
        report_to=script_args.report_to,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_steps=script_args.warmup_steps,
        optim=script_args.optimizer_type,
        bf16=script_args.bf16,
        remove_unused_columns=False,
        run_name=script_args.run_name,
        max_grad_norm=script_args.max_grad_norm,
        deepspeed=script_args.deepspeed,
        num_train_epochs=script_args.num_train_epochs,
        save_strategy=script_args.save_strategy,
        save_total_limit=script_args.save_total_limit,
        warmup_ratio=script_args.warmup_ratio,
        tf32=script_args.tf32,
        dataloader_num_workers=script_args.dataloader_num_workers,
        fp16=script_args.fp16,
        seed=script_args.seed,
    )

    
    # initialize the DPO trainer
    dpo_trainer = LlavaDPOTrainer(
        model=llava_policy_model,
        ref_model=llava_ref_model,
        args=training_args,
        beta=script_args.beta,
        tokenizer=tokenizer,
        max_prompt_length=script_args.max_prompt_length,
        max_length=script_args.max_length,
        **data_module,
    )
    
    dpo_trainer.add_callback(SaverCallback())
    
    dpo_trainer.train(resume_from_checkpoint=script_args.resume_from_checkpoint)
    
if __name__ == "__main__":
    main()