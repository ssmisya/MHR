
target_lang=$1

# export SLURM_JOB_ID=2723316
gpus=1
cpus=16
quotatype="reserved"
OMP_NUM_THREADS=4 srun --partition=MoE --job-name="tranlate" --mpi=pmi2 --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype} \
python /mnt/petrelfs/songmingyang/code/mm/MAPO/m3apo/preprocess/translate.py \
--model_path /mnt/petrelfs/songmingyang/songmingyang/model/others/nllb-200-distilled-600M \
--input_file_path /mnt/petrelfs/songmingyang/songmingyang/data/mm/annotation/hadpo-data/hadpo/llava-v1.5/desc_data.json \
--output_file_path /mnt/petrelfs/songmingyang/songmingyang/data/mm/annotation/hadpo-data/hadpo/llava-v1.5/multilingual/desc_${target_lang}.json \
--target_language ${target_lang} \
--batch_size 128