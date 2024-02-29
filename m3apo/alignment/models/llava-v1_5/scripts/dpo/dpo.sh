source ~/.bashrc
source ~/anaconda3/bin/activate vcd
code_base=/mnt/petrelfs/songmingyang/code/mm/MAPO/m3apo/alignment/models/llava-v1_5/
cd $code_base

seed=55

model_name_or_path=/mnt/petrelfs/songmingyang/songmingyang/model/others/llava-v1.5-7b
dataset_path=/mnt/petrelfs/songmingyang/songmingyang/runs/llava/test/merged/llava_multilingual_dpo_data.jsonl
vision_tower_path=/mnt/petrelfs/songmingyang/songmingyang/model/others/clip-vit-large-patch14-336
image_folder=/mnt/petrelfs/songmingyang/songmingyang/data/mm/imgs/train2017

accelerate_config_file=/mnt/petrelfs/songmingyang/code/mm/MAPO/m3apo/alignment/models/llava-v1_5/scripts/deepspeed/accelerate_config.yaml
ckpt_save_path=/mnt/petrelfs/songmingyang/songmingyang/runs/llava/dpo/checkpoints/llava-v1_5

gpus=4
cpus=64
quotatype="reserved"

OMP_NUM_THREADS=4 srun --partition=MoE --job-name="generate" --mpi=pmi2 --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype} \
accelerate launch --config_file=${accelerate_config_file}  ./train_dpo.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 0 \
    --deepspeed ./scripts/deepspeed/zero3.json \
    --model_name_or_path ${model_name_or_path} \
    --version v1 \
    --data_path ${dataset_path} \
    --vision_tower ${vision_tower_path} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --image_folder ${image_folder} \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ${ckpt_save_path} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
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
    --run_name "llava-v1.5" \
    --beta 0.1