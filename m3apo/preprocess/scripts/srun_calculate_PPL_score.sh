source ~/.bashrc
source ~/anaconda3/bin/activate vcd
# code_base=/mnt/petrelfs/songmingyang/code/mm/MAPO/Preprocess
# cd $code_base

seed=55
preprocess_base=/mnt/petrelfs/songmingyang/code/mm/MAPO/m3apo/preprocess
data_dir_base=/mnt/petrelfs/songmingyang/songmingyang/runs/llava/test
input_data_dir=${data_dir_base}/generations
output_data_dir=${data_dir_base}/add_ppl/20_en_refs
reference_en_file=${data_dir_base}/generations/llava_7b_v1_generation_num20_en.json
reward_model_path=/mnt/petrelfs/songmingyang/songmingyang/model/others/nllb-200-distilled-600M

default_language=en
language=${1:- default_language}
num_proc=1
data_len=10000
for ((i = 0; i < num_proc; i++)); do
    begin_index=$((i * data_len))
    gpus=1
    cpus=16
    quotatype="reserved"
    OMP_NUM_THREADS=4 srun --partition=MoE --job-name="calc ppl" --mpi=pmi2 --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype} \
    python ${preprocess_base}/calculate_PPL_score.py \
    --begin_index "$begin_index" \
    --data_length "$data_len" \
    --data_file llava_7b_v1_generation_num20_${language}.json \
    --reward_model_path $reward_model_path \
    --language $language \
    --reference_style "ref_file" \
    --input_data_dir ${input_data_dir} \
    --output_data_dir ${output_data_dir} 1>${preprocess_base}/scripts/calc_ppl_logs/log_${language}_${begin_index}_${data_len} 2>&1 &  
    
    echo "Process $i Begin!"
    processes+=($!)
done


    # --reference_en_file ${reference_en_file} \
