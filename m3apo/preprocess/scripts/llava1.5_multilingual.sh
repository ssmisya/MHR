#!/usr/bin/bash

#SBATCH --job-name=llava_test
#SBATCH --output=/mnt/petrelfs/songmingyang/songmingyang/runs/llava/test/logs/%x-%j.log
#SBATCH --error=/mnt/petrelfs/songmingyang/songmingyang/runs/llava/test/logs/%x-%j.log

#SBATCH --partition=MoE
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --quotatype=reserved

# spot reserved auto
num_nodes=1      # should match with --nodes
nproc_per_node=1 # should match with --gres

# machine and environment settings
nodes=($(scontrol show hostnames $SLURM_JOB_NODELIS))
nodes_array=($nodes)
head_node=${nodes_array[0]}
echo "Node: $head_node"

export OMP_NUM_THREADS=8

source ~/.bashrc
source ~/anaconda3/bin/activate vcd


code_base=/mnt/petrelfs/songmingyang/code/mm/MAPO/Preprocess
cd $code_base

seed=55
dataset_name=coco
type=adversarial
model_path=/mnt/petrelfs/songmingyang/songmingyang/model/others/llava-v1.5-7b
cd_alpha=-1
cd_beta=0.2
noise_step=-500
generation_num=5
language=fr

image_folder=/mnt/petrelfs/songmingyang/songmingyang/data/mm/imgs/train2017
question_file=/mnt/petrelfs/songmingyang/songmingyang/data/mm/annotation/LLaVA-Human-Preference-10K/llava_7b_v1_preference.json
generation_dir_path=/mnt/petrelfs/songmingyang/songmingyang/runs/llava/test/generations
generation_file=${generation_dir_path}/llava_7b_v1_generation_num${generation_num}_${language}.json

python ./lvlm_sampling.py \
--model-path ${model_path} \
--question-file  ${question_file} \
--image-folder ${image_folder} \
--answers-file  ${generation_file}\
--answers_file_format "json" \
--cd_alpha $cd_alpha \
--cd_beta $cd_beta \
--noise_step $noise_step \
--seed ${seed} \
--language ${language} \
--generation_num ${generation_num} 

# --use_cd \

