source ~/.bashrc
source ~/anaconda3/bin/activate vcd
# code_base=/mnt/petrelfs/songmingyang/code/mm/MAPO/Preprocess
# cd $code_base

seed=55
preprocess_base=/mnt/petrelfs/songmingyang/code/mm/MAPO/m3apo/preprocess

reward_model_path=/mnt/petrelfs/songmingyang/songmingyang/model/others/nllb-200-distilled-600M
bleu_home=/mnt/petrelfs/songmingyang/songmingyang/runs/llava/preprocess/sft_on_palo_1000/calc_score/bleu/results


default_language=fr
language=${1:- $default_language}
alignment_strategy=${2:-"language"}

input_data_dir=/mnt/petrelfs/songmingyang/songmingyang/runs/llava/preprocess/sft_on_palo_1000/desc_generations
file_name=llava_sft_palo_generation_vg_num20_${language}.json

if [ $alignment_strategy == "language" ]; then
    output_data_dir=/mnt/petrelfs/songmingyang/songmingyang/runs/llava/preprocess/sft_on_palo_1000/desc_language_add_ppl
    reference_en_file=/mnt/petrelfs/songmingyang/songmingyang/runs/llava/preprocess/sft_on_palo_1000/desc_generations/llava_sft_palo_generation_vg_num20_en.json
    
elif [ $alignment_strategy == "preference" ]; then
    output_data_dir=/mnt/petrelfs/songmingyang/songmingyang/runs/llava/preprocess/sft_on_palo_1000/desc_preference_add_ppl
    reference_en_file="fuck"
else
    output_data_dir=/mnt/petrelfs/songmingyang/songmingyang/runs/llava/preprocess/sft_on_palo_1000/desc_hallucination_add_ppl
    reference_en_file="fuck"
fi


gpus=1
cpus=16
quotatype="reserved"
OMP_NUM_THREADS=4 srun --partition=MoE --job-name="calc ppl" --mpi=pmi2 --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype} \
python ${preprocess_base}/desc_calculate_ppl_score.py \
    --data_file ${file_name} \
    --reward_model_path $reward_model_path \
    --language $language \
    --alignment_strategy ${alignment_strategy} \
    --reference_en_file ${reference_en_file} \
    --input_data_dir ${input_data_dir} \
    --output_data_dir ${output_data_dir}   
    



    # --reference_en_file ${reference_en_file} \
