cfg_file=/mnt/petrelfs/songmingyang/code/tools/LAVIS/lavis/projects/ft_instruct_blip/qa_palo_vicuna7b_train.yaml
train_code_file=/mnt/petrelfs/songmingyang/code/tools/LAVIS/train.py

export SLURM_JOB_ID=2659664
AD_NAME=songmingyang
AD_PASSWORD=959291Aa
export http_proxy=http://${AD_NAME}:${AD_PASSWORD}@10.1.8.50:33128/
export https_proxy=http://${AD_NAME}:${AD_PASSWORD}@10.1.8.50:33128/ 
export HTTP_PROXY=http://${AD_NAME}:${AD_PASSWORD}@10.1.8.50:33128/ 
export HTTPS_PROXY=http://${AD_NAME}:${AD_PASSWORD}@10.1.8.50:33128/
export HF_ENDPOINT=https://hf-mirror.com

gpus=8
cpus=64
quotatype="reserved"
OMP_NUM_THREADS=8 srun --partition=MoE --job-name="eval" --mpi=pmi2  --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype}  \
python -m torch.distributed.run --nproc_per_node=8 $train_code_file --cfg-path $cfg_file