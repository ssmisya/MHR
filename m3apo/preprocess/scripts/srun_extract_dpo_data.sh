source ~/.bashrc
source ~/anaconda3/bin/activate vcd

seed=55

default_language=en
language=${1:- default_language}

gpus=1
cpus=16
quotatype="reserved"
OMP_NUM_THREADS=4 srun --partition=MoE --job-name="generate" --mpi=pmi2 --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype} \
python /mnt/petrelfs/songmingyang/code/mm/MAPO/m3apo/preprocess/extract_lvlm_dpo_data.py \
--file_name llava_7b_v1_generation_num20_${language}.json_0_2000.jsonl \
--language $language 