import argparse
import torch
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
from mhr.vcd.experiments.llava.utils import disable_torch_init
from mhr.vcd.experiments.llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from vcd_utils.vcd_add_noise import add_diffusion_noise
from vcd_utils.vcd_sample import evolve_vcd_sampling
evolve_vcd_sampling()
from mhr.vcd.experiments.eval.language_dict import language_dict

from mhr.utils.utils import process_jsonl

from PIL import Image
import math

# import kornia
from transformers import set_seed
import random

import debugpy

language_list=["en","fr","ru","ja","th","sw","bn"]

def load_json_file(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data

def construct_prompt_and_inference(model,qs, tokenizer,image_processor,image_file,args):
        cur_prompt = qs
        if not DEFAULT_IMAGE_TOKEN in qs:
            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        # prompt_suffix = language_dict[args.language]['prompt_suffix'].format(language_dict[args.language]['yes'],language_dict[args.language]['no'])
        prompt_suffix = f" Please answer this question in {language_dict[args.language]['full_name']}."
        cur_full_prompt = qs + prompt_suffix
        conv.append_message(conv.roles[0], cur_full_prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        image = Image.open(os.path.join(args.image_folder, image_file))
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        
        if args.use_cd:
            image_tensor_cd = add_diffusion_noise(image_tensor, args.noise_step)
        else:
            image_tensor_cd = None      

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).to(torch.bfloat16).cuda(),
                images_cd=(image_tensor_cd.unsqueeze(0).to(torch.bfloat16).cuda() if image_tensor_cd is not None else None),
                cd_alpha = args.cd_alpha,
                cd_beta = args.cd_beta,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_new_tokens=512,
                use_cache=True,
                num_return_sequences=args.generation_num,  # 控制生成的序列数量
                )
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)
        new_outputs = []
        for output in outputs:
            output = output.strip()
            if output.endswith(stop_str):
                output = output[:-len(stop_str)]
            output = output.strip()
            new_outputs.append(output)
        return cur_full_prompt, new_outputs
    

def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    
    answers_file = os.path.expanduser(args.answers_file)
    if os.path.exists(answers_file):
        exist_data = process_jsonl(answers_file)
        ans_file = open(answers_file, "a+",encoding="utf8")
    else:
        os.makedirs(os.path.dirname(answers_file), exist_ok=True)
        ans_file = open(answers_file, "a+",encoding="utf8")
        exist_data = None
    
    if args.dataset_type == "vg":
        vg_image_data = json.load(open(os.path.join(args.vg_path, "image_data.json")))
        id2path = {
            _data["image_id"]:os.path.join(args.image_folder, _data["url"].split("/")[-2], _data["url"].split("/")[-1]) 
            for _data in vg_image_data
        }
        questions = json.load(open(args.question_file, "r",encoding="utf8"))
        k=0
        for image_id in tqdm(questions.keys()):
            if exist_data is not None and k < len(exist_data) and str(image_id) == exist_data[k]["question_id"]: 
                k+=1
                continue
            image_file = id2path[int(image_id)]
            idx=image_id
            qs = random.choice([
                        "Describe this image in detail.",
                        "Take a look at this image and describe what you notice.",
                        "Please provide a detailed description of the picture.",
                        "Could you describe the contents of this image for me?",
                    ])
            cur_prompt, new_outputs = construct_prompt_and_inference(model,qs, tokenizer,image_processor,image_file,args)
            ans_file.write(json.dumps({"question_id": idx,
                                    "prompt": cur_prompt,
                                    "answer": new_outputs,
                                    "model_id": model_name,
                                    "image": image_file,
                                    "en_answer":questions[image_id]["chosen"],
                                    "reject_en_answer":questions[image_id]["rejected"],
                                    "language":args.language,
                                    "metadata": {}},ensure_ascii=False) + "\n")
            ans_file.flush()
    
    elif args.dataset_type == "human_preference_10k":

        if args.question_file_format == "jsonl":
            questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
        else:
            questions = load_json_file(args.question_file)

        k=0
        for line in tqdm(questions):
            if exist_data is not None and k < len(exist_data) and line["id"] == exist_data[k]["question_id"]:
                k+=1
                continue
            
            if args.question_file_format == "jsonl":
                idx = line["question_id"]
                image_file = line["image"]
                qs = line["text"]
            else:
                idx = line['id']
                image_file = line['image']
                question_dict = line['conversations'][-2]
                assert question_dict['from'] == 'human'
                qs = question_dict['value']
            
            cur_prompt, new_outputs = construct_prompt_and_inference(model,qs, tokenizer,image_processor,image_file,args)
            ans_file.write(json.dumps({"question_id": idx,
                                    "prompt": cur_prompt,
                                    "answer": new_outputs,
                                    "model_id": model_name,
                                    "image": image_file,
                                    "en_answer":line[f"output_{line['preference']}"]['value'],
                                    "language":args.language,
                                    "metadata": {}},ensure_ascii=False) + "\n")
            ans_file.flush()
    else:
        raise ValueError(f"dataset_type {args.dataset_type} not supported")
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image_folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--question_file_format", type=str, default="jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--answers_file_format", type=str, default="jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=None)

    parser.add_argument("--noise_step", type=int, default=500)
    parser.add_argument("--use_cd", action='store_true', default=False)
    parser.add_argument("--cd_alpha", type=float, default=1)
    parser.add_argument("--cd_beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--generation_num", type=int, default=5)
    parser.add_argument("--dataset_type",type=str,default="vg")
    parser.add_argument("--vg_path",type=str,default="/mnt/petrelfs/songmingyang/songmingyang/data/mm/annotation/vg")
    
    args = parser.parse_args()
    set_seed(args.seed)
    eval_model(args)


