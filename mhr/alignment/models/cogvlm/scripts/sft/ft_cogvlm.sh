#!/usr/bin/bash
#SBATCH --job-name=train_cogvlm
#SBATCH --output=/mnt/petrelfs/songmingyang/songmingyang/runs/robustlmm/baseline/pretrain/logs/%x-%j.log
#SBATCH --error=/mnt/petrelfs/songmingyang/songmingyang/runs/robustlmm/baseline/pretrain/logs/%x-%j.log

#SBATCH --partition=MoE
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0

#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --quotatype=reserved

# spot reserved auto
num_nodes=1      # should match with --nodes
gpus=8           # should match with --gres
cpus=64         # should match with --cpus-per-task
quotatype="reserved"

# basic settings
# nodes=($(scontrol show hostnames $SLURM_JOB_NODELIS))
# nodes_array=($nodes)
# head_node=${nodes_array[0]}
# echo "Node: $head_node"

# environment variables
export OMP_NUM_THREADS=4
AD_NAME=songmingyang
AD_PASSWORD=959291Aa
export http_proxy=http://${AD_NAME}:${AD_PASSWORD}@10.1.8.50:33128/
export https_proxy=http://${AD_NAME}:${AD_PASSWORD}@10.1.8.50:33128/ 
export HTTP_PROXY=http://${AD_NAME}:${AD_PASSWORD}@10.1.8.50:33128/ 
export HTTPS_PROXY=http://${AD_NAME}:${AD_PASSWORD}@10.1.8.50:33128/
export HF_ENDPOINT=https://hf-mirror.com

### CUDA settings
export PATH=/mnt/petrelfs/share/gcc/gcc-11.2.0/bin:$PATH
export CUDA_HOME=/mnt/petrelfs/share/cuda-11.8
export PATH=/mnt/petrelfs/share/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/mnt/petrelfs/share/cuda-12.1/lib64:$LD_LIBRARY_PATH

# virtual environment
source ~/.bashrc
source ~/anaconda3/bin/activate smoe
code_base=/mnt/petrelfs/songmingyang/code/mm/MAPO/mhr/alignment/models/cogvlm/CogVLM/finetune_demo
cd $code_base


# model & data path
vicuna_path=/mnt/petrelfs/songmingyang/songmingyang/model/others/vicuna-7b-v1.5

# data path
data_annotation_path=/mnt/petrelfs/songmingyang/songmingyang/data/mm/annotation/palo_train_data/palo_multilingual_dataset.json
image_dir_path=/mnt/petrelfs/songmingyang/songmingyang/data/mm/imgs

#ckpt save path
ckpt_save_dir=/mnt/petrelfs/songmingyang/songmingyang/runs/cogvlm/sft/ckpt
ckpt_name=cogvlm-chat-v1.1
ckpt_save_path=${ckpt_save_dir}/${ckpt_name}
mkdir -p ${ckpt_save_path}

# config path
zero2_config=/mnt/petrelfs/songmingyang/code/mm/MAPO/mhr/alignment/models/cogvlm/CogVLM/finetune_demo/sft_palo_bf16.json
export SLURM_JOB_ID=3229921  
# unset SLURM_JOB_ID




NUM_GPUS_PER_WORKER=8
MP_SIZE=1

# script_path=$(realpath $0)
# script_dir=$(dirname $script_path)
# main_dir=$(dirname $script_dir)
MODEL_TYPE="cogvlm-chat-v1.1"
VERSION="chat"
MODEL_ARGS="--from_pretrained $MODEL_TYPE \
    --max_length 2048 \
    --lora_rank 10 \
    --use_lora \
    --local_tokenizer ${vicuna_path} \
    --version $VERSION"
# Tips: If training models of resolution 244, you can set --max_length smaller 

OPTIONS_SAT="SAT_HOME=~/.sat_models"
OPTIONS_NCCL="NCCL_DEBUG=info NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2 LOCAL_WORLD_SIZE=$NUM_GPUS_PER_WORKER"
HOST_FILE_PATH="hostfile"

train_data="./archive_split/train"
valid_data="./archive_split/valid"

gpt_options=" \
       --experiment-name finetune-$MODEL_TYPE \
       --model-parallel-size ${MP_SIZE} \
       --mode finetune \
       --train-iters 5000 \
       --resume-dataloader \
       $MODEL_ARGS \
       --train-data ${train_data} \
       --valid-data ${valid_data} \
       --distributed-backend nccl \
       --lr-decay-style cosine \
       --warmup .02 \
       --checkpoint-activations \
       --vit_checkpoint_activations \
       --save-interval 1000 \
       --eval-interval 99999999999 \
       --save ${ckpt_save_path} \
       --eval-iters 10 \
       --eval-batch-size 1 \
       --split 1. \
       --deepspeed_config ${zero2_config} \
       --skip-init \
       --seed 2023 \
       --image_dir_path ${image_dir_path} \
       --data_annotation_path ${data_annotation_path} \
"

              

run_cmd=" deepspeed --master_port 16666 --hostfile ${HOST_FILE_PATH} finetune_cogvlm_demo.py ${gpt_options}"
echo ${run_cmd}

# ${OPTIONS_NCCL} ${OPTIONS_SAT} 
NCCL_DEBUG=info NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2 LOCAL_WORLD_SIZE=$NUM_GPUS_PER_WORKER SAT_HOME=~/.sat_models \
srun --partition=MoE --job-name="train_cogvlm" --mpi=pmi2  --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype}  \
${run_cmd}

set +x