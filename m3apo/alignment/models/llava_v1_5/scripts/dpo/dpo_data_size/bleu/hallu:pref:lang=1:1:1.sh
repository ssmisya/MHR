#!/usr/bin/bash

#SBATCH --job-name=dpo
#SBATCH --output=/mnt/petrelfs/songmingyang/songmingyang/runs/llava/dpo/logs/%x-%j.log
#SBATCH --error=/mnt/petrelfs/songmingyang/songmingyang/runs/llava/dpo/logs/%x-%j.log

#SBATCH --partition=MoE
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0

#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --quotatype=reserved

# spot reserved auto
num_nodes=1      # should match with --nodes
nproc_per_node=8 # should match with --gres
cpus=64
quotatype=reserved

unset SLURM_JOB_ID

# environment variables
export OMP_NUM_THREADS=$nproc_per_node
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
source ~/anaconda3/bin/activate vcd_origin
code_base=/mnt/petrelfs/songmingyang/code/mm/MAPO/m3apo/alignment/models/llava_v1_5/
cd $code_base

seed=55

# program settings
accelerate_config_file=/mnt/petrelfs/songmingyang/code/mm/MAPO/m3apo/alignment/models/llava_v1_5/scripts/deepspeed/accelerate_config.yaml
ckpt_save_dir=/mnt/petrelfs/songmingyang/songmingyang/runs/llava/dpo/checkpoints
vision_tower_path=/mnt/petrelfs/songmingyang/songmingyang/model/others/clip-vit-large-patch14-336
model_name_or_path=/mnt/petrelfs/songmingyang/songmingyang/model/mm/ckpts/sft_palo/llava_checkpoint_1000

ckpt_name=bleu_hallu_pref_lang_eq_1_1_1
ckpt_save_path=${ckpt_save_dir}/${ckpt_name}
mkdir -p ${ckpt_save_path}
# Data paths
hallucination_data=/mnt/petrelfs/songmingyang/songmingyang/runs/llava/preprocess/sft_on_palo_1000/desc_dpo_data/bleu/hallucination
language_data=/mnt/petrelfs/songmingyang/songmingyang/runs/llava/preprocess/sft_on_palo_1000/desc_dpo_data/bleu/language
preference_data=/mnt/petrelfs/songmingyang/songmingyang/runs/llava/preprocess/sft_on_palo_1000/desc_dpo_data/bleu/preference

image_folder=/mnt/petrelfs/songmingyang/songmingyang/data/mm/imgs/vg
save_steps=1000
resume_from_checkpoint="True"

srun --partition=MoE --job-name="dpo" --mpi=pmi2 --gres=gpu:${nproc_per_node} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype} \
accelerate launch --config_file=${accelerate_config_file}  ./train_dpo.py \
    --deepspeed ./scripts/deepspeed/zero3.json \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 0 \
    --model_name_or_path ${model_name_or_path} \
    --version v1 \
    --vision_tower ${vision_tower_path} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ${ckpt_save_path} \
    --num_train_epochs 9 \
    --per_device_train_batch_size 8 \
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
    --report_to wandb \
    --run_name ${ckpt_name} \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --beta 0.1 \
    --hallucination_data_path ${hallucination_data} \
    --hallucination_data_type "dir_of_jsonl_desc" \
    --hallucination_ratio 1 \
    --language_data_path ${hallucination_data} \
    --language_data_type "dir_of_jsonl_desc" \
    --language_ratio 1 \
    --preference_data_path ${preference_data} \
    --preference_ratio 1 \
    --preference_data_type "dir_of_jsonl_desc" \
    --image_folder ${image_folder} \
    --vg_path /mnt/petrelfs/songmingyang/songmingyang/data/mm/annotation/vg \
    --resume_from_checkpoint ${resume_from_checkpoint}


   
    # --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 0 \
        # --data_path ${dataset_path} \
            # --image_folder ${image_folder} \