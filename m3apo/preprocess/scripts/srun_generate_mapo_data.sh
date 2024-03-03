
source ~/.bashrc
source ~/anaconda3/bin/activate vcd


code_base=/mnt/petrelfs/songmingyang/code/mm/MAPO/m3apo/preprocess
cd $code_base

seed=55

$default_language='bg'
language=${1:-default_language}
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:4096
# export CUDA_DEVICE_ORDER="PCI_BUS_ID"

accelerate_config_file=/mnt/petrelfs/songmingyang/code/mm/MAPO/m3apo/alignment/models/llava-v1_5/scripts/deepspeed/4gpus_wo_deepspeed.yaml

gpus=4
cpus=32
quotatype="reserved"
OMP_NUM_THREADS=4 srun --partition=MoE --job-name="generate" --mpi=pmi2 --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype} \
accelerate launch --config_file=$accelerate_config_file ./generate_mapo_data.py --language $language