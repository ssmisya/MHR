#!/usr/bin/bash

#SBATCH --job-name=llava
#SBATCH --output=/mnt/petrelfs/songmingyang/songmingyang/runs/llava/logs/%x-%j.log
#SBATCH --error=/mnt/petrelfs/songmingyang/songmingyang/runs/llava/logs/%x-%j.log

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

# AD_NAME=songmingyang
# AD_PASSWORD=959291Aa
# export http_proxy=http://${AD_NAME}:${AD_PASSWORD}@10.1.8.50:33128/
# export https_proxy=http://${AD_NAME}:${AD_PASSWORD}@10.1.8.50:33128/ 
# export HTTP_PROXY=http://${AD_NAME}:${AD_PASSWORD}@10.1.8.50:33128/ 
# export HTTPS_PROXY=http://${AD_NAME}:${AD_PASSWORD}@10.1.8.50:33128/
# export HF_ENDPOINT=https://hf-mirror.com

seed=55
dataset_name=coco
type=adversarial
model_path=/mnt/petrelfs/songmingyang/songmingyang/model/others/llava-v1.5-7b
cd_alpha=-1
cd_beta=0.2
noise_step=-500
language=en
if [[ $dataset_name == 'coco' || $dataset_name == 'aokvqa' ]]; then
  image_folder=/mnt/petrelfs/share_data/quxiaoye/VCD_file/val2014
else
  image_folder=./data/gqa/images
fi

code_base=/mnt/petrelfs/songmingyang/code/VCD/experiments
cd $code_base

python ./eval/object_hallucination_vqa_llava.py \
--model-path ${model_path} \
--question-file ./data/POPE/multi_lingual/${dataset_name}/${language}/${dataset_name}_pope_${type}_${language}.json \
--image-folder ${image_folder} \
--answers-file ./output/llava15_${dataset_name}_pope_${type}_answers_no_cd_seed${seed}_${language}.jsonl \
--cd_alpha $cd_alpha \
--cd_beta $cd_beta \
--noise_step $noise_step \
--seed ${seed} \
--language ${language}

# --use_cd \

