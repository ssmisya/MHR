import os
import time
import json
import argparse
import subprocess
from pathlib import Path
from typing import Tuple, List

from loguru import logger
import queue
import threading
from concurrent.futures import ThreadPoolExecutor

def run_command(command):
    try:
        logger.info(f"Running cmd: {command}")
        subprocess.run(
            command,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"An error occurred: {e}")

def main():
    parser = argparse.ArgumentParser(description='Your description here')
    parser.add_argument('--use_cd', action='store_true', help='Use CD if specified')

    args = parser.parse_args()
    languages = "en ar  bg  bn  de  el  es  fa  fr  gu  hi  id  it  ja  jv  ko  ml  mr  ms  my  nl  pt  ru  sv  sw  ta  te  th  tr  uk  ur  vi  zh"
    # languages = "en bn fr ru th sw ja"
    language_list = languages.split()
    dataset_list=["coco"]
    type_list=["popular"]
    seed=55
    cd_alpha=-1
    cd_beta=0.2
    noise_step=-500
    partition="MoE"
    model_path="/mnt/petrelfs/songmingyang/songmingyang/model/others/llava-v1.5-7b"
    # model_path="/mnt/petrelfs/songmingyang/songmingyang/model/mm/ckpts/sft_palo/llava_checkpoint_1000"
    # peft_model_path="/mnt/petrelfs/songmingyang/songmingyang/runs/llava/dpo/checkpoints/direct_dpo/origin_human_preference/6000"
    # peft_model_path="/mnt/petrelfs/songmingyang/songmingyang/runs/llava/dpo/checkpoints/direct_dpo/sft_human_preference"
    # peft_model_path="/mnt/petrelfs/songmingyang/songmingyang/runs/llava/dpo/checkpoints/direct_dpo/origin_self_hallucination"
    peft_model_path="/mnt/petrelfs/songmingyang/songmingyang/runs/llava/dpo/checkpoints/direct_dpo/sft_self_hallucination"

    
    image_folder="/mnt/petrelfs/share_data/quxiaoye/VCD_file/val2014"
    vcd_base="/mnt/petrelfs/songmingyang/code/mm/MAPO/m3apo/vcd/experiments"
    
    output_file_base = "/mnt/petrelfs/songmingyang/songmingyang/runs/llava/dpo/generations/direct_dpo"
    output_file_dir = "sft_sh"
    os.makedirs(f"{output_file_base}/{output_file_dir}", exist_ok=True)
    for i in range(len(dataset_list)):
        for j in range(len(type_list)):
            for k in range(len(language_list)):
                dataset_name=dataset_list[i]
                type_item=type_list[j]
                language=language_list[k]
                
                output_file=f"{output_file_base}/{output_file_dir}/llava15_sft_pope_{type_item}_answers_{language}.jsonl"
                if os.path.exists(output_file):
                    continue
                else:
                    cmd_args = [
                            f" --model-path {model_path} ",
                            f" --question-file {vcd_base}/data/POPE/multi_lingual/{dataset_name}/{language}/{dataset_name}_pope_{type_item}_{language}.json ",
                            f" --image-folder {image_folder} ",
                            f" --answers-file {output_file} ",
                            f" --cd_alpha {cd_alpha} ",
                            f" --cd_beta {cd_beta} ",
                            f" --noise_step {noise_step} ",
                            f" --seed {seed}",
                            f" --language {language} ",
                        ]
                    if args.use_cd:
                        cmd_args.append(" --use_cd ")   
                    if peft_model_path:
                        cmd_args.append(f" --peft_model_path {peft_model_path}")
                    log_path = f"/mnt/petrelfs/songmingyang/songmingyang/runs/llava/logs/llava15_{dataset_name}_pope_{type_item}_answers_no_cd_seed{seed}_{language}_{output_file_dir}.log"
                    # commands.append(f"nohup srun -p {partition} -n1 -N1 --gres=gpu:1 --quotatype=auto -c 16 --job-name=generate "
                    #         + f"--output={log_path} "
                    #         + f"--error={log_path} "
                    #         + "python /mnt/petrelfs/songmingyang/code/VCD/experiments/eval/object_hallucination_vqa_llava.py "
                    #         + " ".join(cmd_args)
                    #         + f" 1>{log_path} 2>&1")
                    run_command(f"nohup srun -p {partition} -n1 -N1 --gres=gpu:1 --quotatype=reserved -c 16 --job-name=llavaGen "
                            + f"--output={log_path} "
                            + f"--error={log_path} "
                            + f"python {vcd_base}/eval/object_hallucination_vqa_llava.py "
                            + " ".join(cmd_args)
                            + f" 1>{log_path} 2>&1 &")
                
    # with ThreadPoolExecutor(max_workers=32) as executor:
    #     executor.map(run_command, commands)
    

if __name__ == '__main__':
    
    main()