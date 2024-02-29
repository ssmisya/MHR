#!/usr/bin/bash

#SBATCH --job-name=llava
#SBATCH --output=/mnt/petrelfs/songmingyang/songmingyang/runs/llava/eval/logs/%x-%j.log
#SBATCH --error=/mnt/petrelfs/songmingyang/songmingyang/runs/llava/eval/logs/%x-%j.log

#SBATCH --partition=MoE
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0

#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --quotatype=reserved

# spot reserved auto
num_nodes=1      # should match with --nodes
nproc_per_node=0 # should match with --gres

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

dataset=( coco )
type=(adversarial popular random)
language=("es" "de" "en" "fr" "it" "pt" "ru" "ja" "ko" "zh" "ar")

for i in ${dataset[@]}; do
  for j in ${type[@]}; do
    for k in ${language[@]}; do
      echo "Evaluating ${i} ${j} ${k}"
      python /mnt/petrelfs/songmingyang/code/VCD/experiments/eval/eval_pope.py \
      --gt_files /mnt/petrelfs/songmingyang/code/VCD/experiments/data/POPE/multi_lingual/${i}/${k}/${i}_pope_${j}_${k}.json \
      --gen_files /mnt/petrelfs/songmingyang/code/VCD/experiments/output/llava15_${i}_pope_${j}_answers_no_cd_seed55_${k}.jsonl \
      --language $k
    done
  done
done

