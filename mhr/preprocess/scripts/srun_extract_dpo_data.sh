source ~/.bashrc
source ~/anaconda3/bin/activate vcd

seed=55

default_language=en
language=${1:- default_language}
alignment_strategy=${2:- "language"}
select_k=${3:- 3}
metric="loss"
metric_home=/mnt/petrelfs/songmingyang/songmingyang/runs/llava/preprocess/sft_on_palo_1000/calc_score/loss

if [ $alignment_strategy == "language" ]; then
    input_data_dir=${metric_home}/desc_language_add_ppl
    extract_method=extract_top3
    
elif [ $alignment_strategy == "preference" ]; then
    input_data_dir=${metric_home}/desc_preference_add_ppl
    extract_method=extract_top3
else
    input_data_dir=${metric_home}/desc_hallucination_add_ppl
    extract_method=extract_self_hallucinagion_top3 
fi

output_data_dir=/mnt/petrelfs/songmingyang/songmingyang/runs/llava/preprocess/sft_on_palo_1000/desc_dpo_data/select_k/top${select_k}/${alignment_strategy}
mkdir -p $output_data_dir
file_name=llava_sft_palo_generation_vg_num20_${language}_${alignment_strategy}.jsonl

gpus=0
cpus=8
quotatype="reserved"
OMP_NUM_THREADS=4 srun --partition=MoE --job-name="extract" --mpi=pmi2 --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype} \
python /mnt/petrelfs/songmingyang/code/mm/MAPO/mhr/preprocess/desc_extract_dpo_data.py \
-i ${input_data_dir} \
-o ${output_data_dir} \
--file_name ${file_name} \
--language $language \
--extract_method $extract_method \
--select_k $select_k