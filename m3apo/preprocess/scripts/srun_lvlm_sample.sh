source ~/.bashrc
source ~/anaconda3/bin/activate vcd


code_base=/mnt/petrelfs/songmingyang/code/mm/MAPO/Preprocess
cd $code_base

seed=55
# dataset_name=coco
# type=adversarial
model_path=/mnt/petrelfs/songmingyang/songmingyang/model/others/llava-v1.5-7b
cd_alpha=-1
cd_beta=0.2
noise_step=-500
generation_num=20
default_language=en
language=${1:-default_language}

image_folder=/mnt/petrelfs/songmingyang/songmingyang/data/mm/imgs/train2017
question_file=/mnt/petrelfs/songmingyang/songmingyang/data/mm/annotation/LLaVA-Human-Preference-10K/llava_7b_v1_preference.json
generation_dir_path=/mnt/petrelfs/songmingyang/songmingyang/runs/llava/test/generations
generation_file=${generation_dir_path}/llava_7b_v1_generation_num${generation_num}_${language}.json

gpus=1
cpus=16
quotatype="reserved"
OMP_NUM_THREADS=4 srun --partition=MoE --job-name="generate" --mpi=pmi2 --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype} \
python ./lvlm_sampling.py \
--model-path ${model_path} \
--question-file  ${question_file} \
--question_file_format "json" \
--image-folder ${image_folder} \
--answers-file  ${generation_file} \
--answers_file_format "json" \
--cd_alpha $cd_alpha \
--cd_beta $cd_beta \
--noise_step $noise_step \
--seed ${seed} \
--language ${language} \
--generation_num ${generation_num} 