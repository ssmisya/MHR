source ~/.bashrc
source ~/anaconda3/bin/activate vcd
# code_base=/mnt/petrelfs/songmingyang/code/mm/MAPO/Preprocess
# cd $code_base

seed=55
preprocess_base=/mnt/petrelfs/songmingyang/code/mm/MAPO/m3apo/preprocess
data_dir_base=/mnt/petrelfs/songmingyang/songmingyang/runs/llava/test

reward_model_path=/mnt/petrelfs/songmingyang/songmingyang/model/others/nllb-200-distilled-600M

default_language=fr
language=${1:- $default_language}
dataset_type=${2:-"vg"}

if [ $dataset_type == "vg" ]; then
    input_data_dir=/mnt/petrelfs/songmingyang/songmingyang/runs/llava/preprocess/sft_on_palo_1000/desc_generations
    output_data_dir=/mnt/petrelfs/songmingyang/songmingyang/runs/llava/preprocess/sft_on_palo_1000/desc_add_ppl
    reference_en_file="fuck"
    reference_style="self_hallucination"
    file_name=llava_sft_palo_generation_vg_num20_${language}.json
else
    input_data_dir=/mnt/petrelfs/songmingyang/songmingyang/runs/llava/preprocess/sft_on_palo_1000/human_generations
    output_data_dir=/mnt/petrelfs/songmingyang/songmingyang/runs/llava/preprocess/sft_on_palo_1000/human_add_ppl
    reference_en_file=/mnt/petrelfs/songmingyang/songmingyang/runs/llava/preprocess/sft_on_palo_1000/human_generations/llava_sft_palo_generation_human_preference_10k_num20_en.json
    file_name=llava_sft_palo_generation_human_preference_10k_num20_${language}.json
    reference_style="ref_file"
fi

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
    --data_file ${file_name} \
    --reward_model_path $reward_model_path \
    --language $language \
    --reference_style ${reference_style} \
    --reference_en_file ${reference_en_file} \
    --input_data_dir ${input_data_dir} \
    --output_data_dir ${output_data_dir} 1>${preprocess_base}/scripts/calc_ppl_logs/log_${language}_${dataset_type}_${begin_index}_${data_len} 2>&1 &  
    
    echo "Process $i Begin!"
    processes+=($!)
done


    # --reference_en_file ${reference_en_file} \
