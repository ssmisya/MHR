gpus=0
cpus=16
quotatype=auto # auto spot reserved
OMP_NUM_THREADS=4 srun --partition=MoE --job-name="eval vcd" --mpi=pmi2 --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype} \
python /mnt/petrelfs/songmingyang/code/VCD/experiments/eval/eval_pope.py \
--gt_files $1 \
--gen_files $2 \
--language en
