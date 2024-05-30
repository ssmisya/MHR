source ~/.bashrc
source ~/anaconda3/bin/activate vcd

seed=55

default_language=en
language=${1:- default_language}

gpus=0
cpus=16
quotatype="reserved"
OMP_NUM_THREADS=4 srun --partition=MoE --job-name="extract" --mpi=pmi2 --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype} \
python /mnt/petrelfs/songmingyang/code/mm/MAPO/m3apo/preprocess/extract_lvlm_dpo_data.py \
-i /mnt/petrelfs/songmingyang/songmingyang/runs/llava/preprocess/sft_on_palo_1000/human_add_ppl \
-o /mnt/petrelfs/songmingyang/songmingyang/runs/llava/preprocess/sft_on_palo_1000/human_dpo_data \
--file_name llava_sft_palo_generation_human_preference_10k_num20_${language}_0_10000.jsonl \
--language $language \
--extract_method extract_top3 