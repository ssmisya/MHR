#!/usr/bin/bash

#SBATCH --job-name=llava_sft_mtlgl
#SBATCH --output=/mnt/petrelfs/songmingyang/songmingyang/runs/llava/sft/logs/%x-%j.log
#SBATCH --error=/mnt/petrelfs/songmingyang/songmingyang/runs/llava/sft/logs/%x-%j.log

#SBATCH --partition=MoE
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0

#SBATCH --nodes=4
#SBATCH --gres=gpu:8
#SBATCH --quotatype=reserved

# spot reserved auto
num_nodes=4     # should match with --nodes
nproc_per_node=8 # should match with --gres


nodes=($(scontrol show hostnames $SLURM_JOB_NODELIS))
nodes_array=($nodes)
head_node=${nodes_array[0]}
echo "Node: $head_node"

export OMP_NUM_THREADS=8

source ~/.bashrc
source ~/anaconda3/bin/activate vcd
AD_NAME=songmingyang
AD_PASSWORD=959291Aa
export http_proxy=http://${AD_NAME}:${AD_PASSWORD}@10.1.8.50:33128/
export https_proxy=http://${AD_NAME}:${AD_PASSWORD}@10.1.8.50:33128/ 
export HTTP_PROXY=http://${AD_NAME}:${AD_PASSWORD}@10.1.8.50:33128/ 
export HTTPS_PROXY=http://${AD_NAME}:${AD_PASSWORD}@10.1.8.50:33128/
export HF_ENDPOINT=https://hf-mirror.com
code_base=/mnt/petrelfs/songmingyang/code/mm/MAPO/m3apo/alignment/models/llava_v1_5/
cd $code_base

model_base=/mnt/petrelfs/songmingyang/songmingyang/model
dataset_base=/mnt/petrelfs/songmingyang/songmingyang/data/mm
run_base=/mnt/petrelfs/songmingyang/songmingyang/runs/llava


model_name_or_path=${model_base}/others/llava-v1.5-7b
vision_tower_path=${model_base}/others/clip-vit-large-patch14-336

data_path=${dataset_base}/annotation/palo_train_data/palo_multilingual_dataset.json
image_folder=${dataset_base}/imgs

deepspeed_config_file=${code_base}/scripts/deepspeed/zero3.json
# accelerate_config_file=${code_base}/scripts/deepspeed/4gpus.yaml
accelerate_config_file=${code_base}scripts/deepspeed/32gpus.yaml
ckpt_save_path=${run_base}/sft/checkpoints/llava_palo_32gpus


run_name=llava-sft-palo
seed=55
save_steps=500

################## VICUNA ##################
PROMPT_VERSION=v1
################## VICUNA ##################

nodes=($(scontrol show hostnames $SLURM_JOB_NODELIS))
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
echo "Node: $head_node"
echo "Node IP: $head_node_ip"
echo "Node list: $SLURM_JOB_NODELIS"
echo "Node Rank: $SLURM_NODEID "


accelerate launch --config_file ${accelerate_config_file} --num_processes 32 --num_machines 4 --main_process_ip ${head_node_ip} --main_process_port 29508 --machine_rank ${SLURM_NODEID} ./train_sft.py \
    --model_name_or_path $model_name_or_path \
    --version $PROMPT_VERSION \
    --data_path ${data_path} \
    --image_folder ${image_folder} \
    --vision_tower ${vision_tower_path} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ${ckpt_save_path} \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 5 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name ${run_name} \
    --image_aspect_ratio 'pad'

    # --pretrain_mm_mlp_adapter ${model_name_or_path}/mm_projector.bin \
        # --deepspeed ${deepspeed_config_file} \
