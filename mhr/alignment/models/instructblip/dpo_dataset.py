import os
import tqdm
import json
import random
from PIL import Image
from omegaconf import OmegaConf
from torch.utils.data import Dataset

# from mhr.alignment.models.instructblip.vigc.common.config import Config
# from mhr.alignment.models.instructblip.vigc.common.registry import registry
import transformers
import torch
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List


class PaloSFTDataset(Dataset):
    """
        AugmentedCaptionDataset:
        use GPT-3.5 augmented revised descriptions as chosen and augmented model response as rejected
    """
    def __init__(self, 
        data_path: str,
        data_args,
        seed: int = 42,
    ):
        super(PaloSFTDataset, self).__init__()
        
        random.seed(seed)
        self.data_path = data_path
        self.data_args = data_args
        self.image_folder = self.data_args.image_folder
        self.vis_processor = self.data_args.image_processor
        self.tokenizer = self.data_args.tokenizer
        self.qformer_tokenizer = self.data_args.qformer_tokenizer
        
        print('Loading SFT Data...')
        self.data = self.preprocess_data(self.data_path)
        print(f"Loaded {len(self.data)} SFT data")
        
        
    
    def preprocess_data(self,data_path):
        dict_list = []
        with open(data_path,"r",encoding="UTF-8") as f:
            data_file = json.load(f)
        for data in data_file:
            if data.get("image",None) is None:
                continue
            image_id = data["id"]
            image_path = data["image"] if os.path.isabs(data["image"]) else os.path.join(self.image_folder,data["image"])
            dict_list.append({
                "image_id": image_id,
                "image_path": image_path,
                "conversations":data["conversations"]
            })
        return dict_list
        
        
    def load_data(self, index):
        anno = self.data[index]
        image_id = anno["image_id"]
        
        image_path = anno["image_path"]
        conversations = anno["conversations"]
        questionAndAnswers = []
        for i in range(0,len(conversations),2):
            if conversations[i]["from"] == "human":
                questionAndAnswers.append((conversations[i]["value"], conversations[i+1]["value"]))
            else:
                raise ValueError("The first message in the conversation should be from the human")
        questionAndAnswersItem = random.choice(questionAndAnswers)[0]
        question = questionAndAnswersItem[0].replace("\n","").replace("<image>","")
        answers = questionAndAnswersItem[1].replace("\n","").replace("<image>","")
        
        encode_item = self.tokenizer(question,return_tensors="pt")
        encode_item_qformer = self.qformer_tokenizer(question,return_tensors="pt")
        input_ids = encode_item["input_ids"]
        attention_mask = encode_item["attention_mask"]
        qfomer_input_ids = encode_item_qformer["input_ids"]
        qfomer_attention_mask = encode_item_qformer["attention_mask"]
        labels = self.tokenizer(answers,return_tensors="pt")["input_ids"]
        
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
                image = expand2square(image, tuple(int(x*255) for x in self.vis_processor.image_mean))
                image = self.vis_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        else:
            image =  self.vis_processor(Image.open(image_path).convert("RGB"),return_tensors="pt")['pixel_values'][0]
        pixel_values = image
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "qfomer_input_ids": qfomer_input_ids,
            "qfomer_attention_mask": qfomer_attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
        }
    
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        sample = self.load_data(index)
        return sample
    
    def collater(self, samples):
        # Filter out None samples
        samples = [s for s in samples if s is not None]
        # Check if samples is empty after filtering
        if not samples:
            return {}
        collated_dict = {}
        keys = samples[0].keys() # Use the keys of the first sample as a reference
        for k in keys:
            values = [sample[k] for sample in samples]
            # If the value type for the key is torch.Tensor, stack them else return list
            collated_dict[k] = torch.stack(values, dim=0) if isinstance(values[0], torch.Tensor) else values
        return collated_dict
    
@dataclass
class DataCollatorForPaloSFTDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    qformer_tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, attention_mask, qformer_input_ids, qformer_attention_mask, labels = tuple([instance[key] for instance in instances]
            for key in ("input_ids", "attention_mask", "qfomer_input_ids", "qfomer_attention_mask", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask,
                                                    batch_first=True,
                                                    padding_value=0)
        qformer_input_ids = torch.nn.utils.rnn.pad_sequence(
            qformer_input_ids,
            batch_first=True,
            padding_value=self.qformer_tokenizer.pad_token_id)
        qformer_attention_mask = torch.nn.utils.rnn.pad_sequence(qformer_attention_mask,
                                                    batch_first=True,
                                                    padding_value=0)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                    batch_first=True,
                                                    padding_value=-100)
        
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        attention_mask = attention_mask[:, :self.tokenizer.model_max_length]
        qformer_input_ids = qformer_input_ids[:, :self.qformer_tokenizer.model_max_length]
        qformer_attention_mask = qformer_attention_mask[:, :self.qformer_tokenizer.model_max_length]
        
        batch = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            qformer_input_ids=qformer_input_ids,
            qformer_attention_mask=qformer_attention_mask,
            labels=labels,
        )

        if 'pixel_values' in instances[0]:
            images = [instance['pixel_values'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['pixel_values'] = torch.stack(images)
            else:
                batch['pixel_values'] = images

        return batch




class AugmentedCaptionDataset(Dataset):
    """
        AugmentedCaptionDataset:
        use GPT-3.5 augmented revised descriptions as chosen and augmented model response as rejected
    """
    def __init__(self, 
        data_path: str,
        vg_path: str,
        cfg: OmegaConf,
        seed: int,
        sample_strategy = "offline",  
    ):
        super(AugmentedCaptionDataset, self).__init__()
        
        random.seed(seed)
        self.data_path = data_path
        self.vg_path = vg_path
        self.cfg = cfg
        
        # pos&neg data
        vis_cfg = self.cfg.datasets.instruct_blip_given_q_coco2017_vig_test.vis_processor.train
        self.vis_processor = self._build_proc_from_cfg(vis_cfg)
        print('Load Description Data...')
        self.data = json.load(open(self.data_path))
        
        self.sample_strategy = sample_strategy
        print(f"sampleing strategy: {self.sample_strategy}")
        assert self.sample_strategy in ["online", "offline"]
        if self.sample_strategy == "offline":
            for index in range(len(self.data)):
                self.data[index]["chosen"] = [random.choice(self.data[index]["chosen"])]
                self.data[index]["rejected"] = [random.choice(self.data[index]["rejected"])]
        
        print(f"Loaded {len(self.data)} description data")
        
        print("Data example:")
        chosen, rejected = self.data[0]["chosen"][0], self.data[0]["rejected"][0]
        print(f"Chosen: {chosen}")
        print(f"Rejected: {rejected}")
        
        # visual genome
        self.vg_image_data = json.load(open(os.path.join(self.vg_path, "image_data.json")))
        self.id2path = {
            _data["image_id"]:os.path.join(self.vg_path, _data["url"].split("/")[-2], _data["url"].split("/")[-1]) 
            for _data in self.vg_image_data
        }
        
    def _build_proc_from_cfg(self, cfg):
        return (
            registry.get_processor_class(cfg.name).from_config(cfg)
            if cfg is not None
            else None
        )
        
    def load_data(self, index):
        anno = self.data[index]
        image_id = anno["image_id"]
        image_path = self.id2path[int(image_id)]
        chosen = random.choice(anno["chosen"])
        rejected = random.choice(anno["rejected"])
        return {
            "image_id": image_id,
            "image_path": image_path,
            "image": self.vis_processor(Image.open(image_path).convert("RGB")),
            "chosen": chosen.replace("\n", ""),
            "rejected": rejected.replace("\n", ""),
            "prompt": random.choice([
                "Describe this image in detail.",
                "Take a look at this image and describe what you notice.",
                "Please provide a detailed description of the picture.",
                "Could you describe the contents of this image for me?",
            ]),
            "data_type": "desc",
        }
        
    def select(self, indices):
        return [
            self.__getitem__(idx) for idx in indices
        ]
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        sample = self.load_data(index)
        return sample
    
    
class PopeDataset(Dataset):
    """
        PopeDataset:
        use GPT-4 reconstructed POPE-format dataset.
    """
    def __init__(self, 
        data_path: str,
        vg_path: str,
        cfg: OmegaConf,
    ):
        super(PopeDataset, self).__init__()
        self.data_path = data_path
        self.vg_path = vg_path
        self.cfg = cfg
        
        # pos&neg data
        vis_cfg = self.cfg.datasets.instruct_blip_given_q_coco2017_vig_test.vis_processor.train
        self.vis_processor = self._build_proc_from_cfg(vis_cfg)
        self.data = json.load(open(self.data_path))
        
        # visual genome
        self.vg_image_data = json.load(open(os.path.join(self.vg_path, "image_data.json")))
        self.id2path = {
            _data["image_id"]:os.path.join(self.vg_path, _data["url"].split("/")[-2], _data["url"].split("/")[-1]) 
            for _data in self.vg_image_data
        }
        self.load_data()
        
    def _build_proc_from_cfg(self, cfg):
        return (
            registry.get_processor_class(cfg.name).from_config(cfg)
            if cfg is not None
            else None
        )
        
    def load_data(self):
        self.data_list = []
        print('Load POPE Data...')
        for anno in tqdm.tqdm(self.data):
            if anno["correct"]:
                continue
            image_id = anno["image_id"]
            image_path = self.id2path[int(image_id)]
            self.data_list.append({
                "image_id": image_id,
                "image_path": image_path,
                "image": self.vis_processor(Image.open(image_path).convert("RGB")),
                "chosen": anno["chosen"],
                "rejected": anno["rejected"],
                "data_type": "pope",
                "prompt": anno["question"],
            })
                
        print(f"Loaded {len(self.data_list)} pope data")
        
        print("Data Example:")
        print("Chosen: ", self.data_list[0]["chosen"])
        print("Rejected: ", self.data_list[0]["rejected"])
        
    def select(self, indices):
        return [
            self.__getitem__(idx) for idx in indices
        ]
        
    def __len__(self):
        return len(self.data_list)
        
    def __getitem__(self, index):
        sample = self.data_list[index]
        return sample