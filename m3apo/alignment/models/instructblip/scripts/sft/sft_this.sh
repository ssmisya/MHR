#!/usr/bin/bash

#SBATCH --job-name=dpo
#SBATCH --output=/mnt/petrelfs/songmingyang/songmingyang/runs/llava/dpo/logs/%x-%j.log
#SBATCH --error=/mnt/petrelfs/songmingyang/songmingyang/runs/llava/dpo/logs/%x-%j.log

#SBATCH --partition=MoE
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0

#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --quotatype=reserved

# spot reserved auto
num_nodes=1      # should match with --nodes
nproc_per_node=4 # should match with --gres

# basic settings
nodes=($(scontrol show hostnames $SLURM_JOB_NODELIS))
nodes_array=($nodes)
head_node=${nodes_array[0]}
echo "Node: $head_node"

# environment variables
export OMP_NUM_THREADS=4
AD_NAME=songmingyang
AD_PASSWORD=959291Aa
export http_proxy=http://${AD_NAME}:${AD_PASSWORD}@10.1.8.50:33128/
export https_proxy=http://${AD_NAME}:${AD_PASSWORD}@10.1.8.50:33128/ 
export HTTP_PROXY=http://${AD_NAME}:${AD_PASSWORD}@10.1.8.50:33128/ 
export HTTPS_PROXY=http://${AD_NAME}:${AD_PASSWORD}@10.1.8.50:33128/
export HF_ENDPOINT=https://hf-mirror.com

export PATH=/mnt/petrelfs/share/gcc/gcc-11.2.0/bin:$PATH

# virtual environment
source ~/.bashrc
source ~/anaconda3/bin/activate vcd
code_base=/mnt/petrelfs/songmingyang/code/mm/MAPO/m3apo/alignment/models/instructblip
cd $code_base

seed=55

# program settings
model_name_or_path=/mnt/petrelfs/songmingyang/songmingyang/model/mm/instruct_blip/instruct_blip_7b
data_path=/mnt/petrelfs/songmingyang/songmingyang/data/mm/annotation/palo_train_data/palo_multilingual_dataset.json
# dataset_path=/mnt/petrelfs/songmingyang/songmingyang/runs/llava/test/merged/llava_multilingual_dpo_data.jsonl
# human_prefer_data_path=/mnt/petrelfs/songmingyang/songmingyang/runs/llava/preprocess/sft_on_palo_1000/human_dpo_data
# self_hallucination_data_path=/mnt/petrelfs/songmingyang/songmingyang/runs/llava/preprocess/sft_on_palo_1000/desc_dpo_data
# vision_tower_path=/mnt/petrelfs/songmingyang/songmingyang/model/others/clip-vit-large-patch14-336
image_folder=/mnt/petrelfs/songmingyang/songmingyang/data/mm/imgs

accelerate_config_file=/mnt/petrelfs/songmingyang/code/mm/MAPO/m3apo/alignment/models/llava_v1_5/scripts/deepspeed/accelerate_config.yaml
ckpt_save_dir=/mnt/petrelfs/songmingyang/songmingyang/runs/instruct_blip/sft/checkpoints

ckpt_name=palo_sfted_human_full
# ckpt_name=sft_self_hallucination_full
ckpt_save_path=${ckpt_save_dir}/${ckpt_name}
mkdir -p ${ckpt_save_path}

save_steps=250
export SLURM_JOB_ID=2659662

accelerate launch --config_file=${accelerate_config_file}  ./train_sft.py \
    --deepspeed /mnt/petrelfs/songmingyang/code/mm/MAPO/m3apo/alignment/models/llava_v1_5/scripts/deepspeed/zero3.json \
    --model_name_or_path ${model_name_or_path} \
    --self_hallucination_data_path ${self_hallucination_data_path} \
    --human_prefer_data_path ${human_prefer_data_path} \
    --image_folder ${image_folder} \
    --data_path ${data_path} \
    --image_aspect_ratio none \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ${ckpt_save_path} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps ${save_steps} \
    --save_total_limit 2 \
    --learning_rate 2e-6 \
    --weight_decay 0. \
    --warmup_steps 0 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --report_to wandb \
    --run_name ${ckpt_name} \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --beta 0.1
