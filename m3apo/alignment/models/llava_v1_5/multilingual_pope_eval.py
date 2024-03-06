import os
import json
import tqdm
import torch
import argparse
import datetime
import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer

from llava.utils import disable_torch_init
from llava.model.builder import load_pretrained_model
from llava.conversation import conv_templates, SeparatorStyle
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from m3apo.vcd.experiments.eval.language_dict import language_dict
from m3apo.utils.utils import process_jsonl

def convert_dict_to_tensor(results, device):
    part_tensor = json.dumps(results)
    part_tensor = torch.Tensor([ord(part_tensor[i]) for i in range(len(part_tensor))]).long().to(device)
    return part_tensor

def evaluate(question_file, output_file, model, image_folder ,tokenizer, image_processor, model_name, args,language="en",conv_mode="llava_v1", ):
    lines = process_jsonl(question_file)
    rank, word_size = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    step = len(lines) // word_size + 1
    start, end = rank * step, (rank + 1) * step
    
    if int(os.environ["RANK"]) == 0:
        print("generating answers...")
    results = []
    for line in tqdm.tqdm(lines[start:end]):
        data = json.loads(line)
        prompt_suffix = language_dict[language]['prompt_suffix'].format(language_dict[language]['yes'],language_dict[language]['no'])
        message_input = data["text"]+prompt_suffix
        # conv.append_message(conv.roles[0], qs + prompt_suffix)
        image = data["image"]
        question_id = data["question_id"]
        
        image = os.path.join(image_folder, image)
        image = Image.open(image).convert("RGB")
        
        conv = conv_templates[conv_mode].copy()
        if "mpt" in model_name.lower():
            roles = ('user', 'assistant')
        else:
            roles = conv.roles

        # Similar operation in model_worker.py
        image_tensor = process_images([image], image_processor, model.config)
        if type(image_tensor) is list:
            image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
        else:
            image_tensor = image_tensor.to(model.device, dtype=torch.float16)

        inp = message_input

        if image is not None:
            # first message
            if model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
            conv.append_message(conv.roles[0], inp)
            image = None
        else:
            # later messages
            conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=False,
                temperature=1.0,
                max_new_tokens=512,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        results.append({
            "question": message_input,
            "text": outputs,
            "question_id": question_id,
        })
        
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
        # res_file = f"pope_{args.set}.jsonl"
        res_file = f"llava15_{args.dataset}_pope_{args.set}_answers_no_cd_{language}.jsonl"
        with open(os.path.join(args.generate_file_save_path, res_file), "w") as f:
            for res in results_all_rank:
                f.write(json.dumps(res,ensure_ascii=False)+'\n')


def main(args):
    # Model
    disable_torch_init()

    local_rank = int(os.environ["LOCAL_RANK"])
    device = f"cuda:{local_rank}"
    model_name = get_model_name_from_path(args.model_path)
    if args.model_base is None:
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=args.model_path, 
            model_base=args.model_base, 
            model_name=model_name,
            load_8bit=args.load_8bit, 
            load_4bit=args.load_4bit, 
            device=device
        )
    else:
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=args.model_path,
            model_base=args.model_base, 
            model_name="llava_lora_model",
            load_8bit=args.load_8bit, 
            load_4bit=args.load_4bit, 
            device=device
        )

    conv_mode = "llava_v1"
    language_path = os.path.join(args.multilingual_pope_path, args.dataset)
    language_list = os.listdir(language_path)
    
    for language in language_list:
        questions_file = os.path.join(language_path, language, f"coco_pope_{args.set}.json")
        evaluate(questions_file, args.generate_file_save_path, model, args.img_path, tokenizer, image_processor, model_name, args, language, conv_mode)
    
        
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    
    parser.add_argument("--pope_path", type=str, required=True)
    parser.add_argument("--set", type=str, required=True)
    
    parser.add_argument("-i","--img_path", type=str, required=True,default="/mnt/petrelfs/share_data/quxiaoye/VCD_file/val2014")
    parser.add_argument("-m","--multilingual_pope_path",type=str,default="/mnt/petrelfs/songmingyang/code/mm/MAPO/m3apo/vcd/experiments/data/POPE/multi_lingual")
    parser.add_argument("-g","--generate_file_save_path", type=str, default="/mnt/petrelfs/songmingyang/code/mm/MAPO/m3apo/vcd/experiments/output/multilingual_gen/test")
    parser.add_argument("-d","--dataset",type=str,default="coco")
    
    args = parser.parse_args()
    
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    args.dist_backend = "nccl"
    print(
        "| distributed init (rank {}, world {}): {}".format(
            int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"]), "env://"
        ),
        flush=True,
    )
    torch.distributed.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=int(os.environ["WORLD_SIZE"]),
        rank=int(os.environ["LOCAL_RANK"]),
        timeout=datetime.timedelta(
            days=365
        ),  # allow auto-downloading and de-compressing
    )
    
    args = parser.parse_args()
    main(args)