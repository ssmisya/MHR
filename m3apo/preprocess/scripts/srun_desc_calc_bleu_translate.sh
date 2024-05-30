
source_lang=$1

# export SLURM_JOB_ID=2723316
gpus=1
cpus=16
quotatype="reserved"
OMP_NUM_THREADS=4 srun --partition=MoE --job-name="tranlate" --mpi=pmi2 --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype} \
python /mnt/petrelfs/songmingyang/code/mm/MAPO/m3apo/preprocess/desc_calc_bleu_translate.py \
--model_path /mnt/petrelfs/songmingyang/songmingyang/model/others/nllb-200-distilled-600M \
--input_file_path /mnt/petrelfs/songmingyang/songmingyang/runs/llava/preprocess/sft_on_palo_1000/desc_generations/llava_sft_palo_generation_vg_num20_${source_lang}.json \
--output_file_path /mnt/petrelfs/songmingyang/songmingyang/runs/llava/preprocess/sft_on_palo_1000/calc_score/bleu/translations/llava_sft_palo_generation_vg_num20_${source_lang}.json \
--target_language en \
--source_language ${source_lang} \
--batch_size 128 \
--input_type jsonl