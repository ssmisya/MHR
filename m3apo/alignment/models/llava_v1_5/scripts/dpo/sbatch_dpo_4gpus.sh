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
code_base=/mnt/petrelfs/songmingyang/code/mm/MAPO/m3apo/alignment/models/llava_v1_5/
cd $code_base

seed=55

# program settings
model_name_or_path=/mnt/petrelfs/songmingyang/songmingyang/model/others/llava-v1.5-7b
# model_name_or_path=/mnt/petrelfs/songmingyang/songmingyang/model/mm/LLaVA-RLHF-7b-v1.5-224/sft_model
# dataset_path=/mnt/petrelfs/songmingyang/songmingyang/runs/llava/test/merged/llava_multilingual_dpo_data.jsonl
self_hallucination_data_path=/mnt/petrelfs/songmingyang/songmingyang/runs/llava/ha_dpo_desc/dpo_data
vision_tower_path=/mnt/petrelfs/songmingyang/songmingyang/model/others/clip-vit-large-patch14-336
image_folder=/mnt/petrelfs/songmingyang/songmingyang/data/mm/imgs/train2017

accelerate_config_file=/mnt/petrelfs/songmingyang/code/mm/MAPO/m3apo/alignment/models/llava-v1_5/scripts/deepspeed/4gpus.yaml
ckpt_save_dir=/mnt/petrelfs/songmingyang/songmingyang/runs/llava/dpo/checkpoints

ckpt_name=origin_self_hallucination_full
# ckpt_name=sft_self_hallucination_full
ckpt_save_path=${ckpt_save_dir}/${ckpt_name}

save_steps=500


accelerate launch --config_file=${accelerate_config_file}  ./train_dpo.py \
    --deepspeed ./scripts/deepspeed/zero3.json \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 0 \
    --model_name_or_path ${model_name_or_path} \
    --version v1 \
    --self_hallucination_data_path ${self_hallucination_data_path} \
    --vision_tower ${vision_tower_path} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ${ckpt_save_path} \
    --num_train_epochs 3 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps ${save_steps} \
    --save_total_limit 5 \
    --learning_rate 2e-6 \
    --weight_decay 0. \
    --warmup_steps 0 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name ${ckpt_name} \
    --beta 0.1
   
    # --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 0 \
        # --data_path ${dataset_path} \
            # --image_folder ${image_folder} \