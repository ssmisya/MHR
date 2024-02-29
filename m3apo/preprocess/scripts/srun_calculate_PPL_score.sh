source ~/.bashrc
source ~/anaconda3/bin/activate vcd
# code_base=/mnt/petrelfs/songmingyang/code/mm/MAPO/Preprocess
# cd $code_base

seed=55

default_language=en
language=${1:- default_language}

gpus=1
cpus=16
quotatype="reserved"
OMP_NUM_THREADS=4 srun --partition=MoE --job-name="generate" --mpi=pmi2 --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype} \
python /mnt/petrelfs/songmingyang/code/mm/MAPO/Preprocess/calculate_PPL_score.py \
--data_file llava_7b_v1_generation_num20_${language}.json \
--begin_index 0 \
--data_length 2000 \
--language $language 

