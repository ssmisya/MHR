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
    language_list = languages.split()
    dataset_list=["coco"]
    type_list=["popular","random","adversarial"]
    seed=55
    cd_alpha=-1
    cd_beta=0.2
    noise_step=-500
    partition="MoE"
    model_path="/mnt/petrelfs/songmingyang/songmingyang/model/others/llava-v1.5-7b"
    image_folder="/mnt/petrelfs/share_data/quxiaoye/VCD_file/val2014"
    commands=[]
    
    for i in range(len(dataset_list)):
        for j in range(len(type_list)):
            for k in range(len(language_list)):
                dataset_name=dataset_list[i]
                type_item=type_list[j]
                language=language_list[k]
                if args.use_cd:
                    output_file=f"/mnt/petrelfs/songmingyang/code/VCD/experiments/output/cd/llava15_{dataset_name}_pope_{type_item}_answers_w_cd_seed{seed}_{language}.jsonl"
                else:
                    output_file=f"/mnt/petrelfs/songmingyang/code/VCD/experiments/output/llava15_{dataset_name}_pope_{type_item}_answers_no_cd_seed{seed}_{language}.jsonl"
                if os.path.exists(output_file):
                    continue
                else:
                    cmd_args = [
                            f" --model-path {model_path} ",
                            f" --question-file /mnt/petrelfs/songmingyang/code/VCD/experiments/data/POPE/multi_lingual/{dataset_name}/{language}/{dataset_name}_pope_{type_item}_{language}.json ",
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
                    log_path = f"/mnt/petrelfs/songmingyang/songmingyang/runs/llava/logs/cd/llava15_{dataset_name}_pope_{type_item}_answers_no_cd_seed{seed}_{language}.log"
                    # commands.append(f"nohup srun -p {partition} -n1 -N1 --gres=gpu:1 --quotatype=auto -c 16 --job-name=generate "
                    #         + f"--output={log_path} "
                    #         + f"--error={log_path} "
                    #         + "python /mnt/petrelfs/songmingyang/code/VCD/experiments/eval/object_hallucination_vqa_llava.py "
                    #         + " ".join(cmd_args)
                    #         + f" 1>{log_path} 2>&1")
                    run_command(f"nohup srun -p {partition} -n1 -N1 --gres=gpu:1 --quotatype=reserved -c 16 --job-name=generate --output=/mnt/petrelfs/songmingyang/songmingyang/runs/llava/logs/cd/%x-%j.log --error=/mnt/petrelfs/songmingyang/songmingyang/runs/llava/logs/cd/%x-%j.log "
                            + f"--output={log_path} "
                            + f"--error={log_path} "
                            + "python /mnt/petrelfs/songmingyang/code/VCD/experiments/eval/object_hallucination_vqa_llava.py "
                            + " ".join(cmd_args)
                            + f" 1>{log_path} 2>&1 &")
                
    # with ThreadPoolExecutor(max_workers=32) as executor:
    #     executor.map(run_command, commands)
    

if __name__ == '__main__':
    
    main()