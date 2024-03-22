source ~/.bashrc
source ~/anaconda3/bin/activate vcd

seed=55
model_path=/mnt/petrelfs/songmingyang/songmingyang/model/mm/ckpts/sft_palo/checkpoint-1000
cd_alpha=-1
cd_beta=0.2
noise_step=-500
generation_num=20
default_language=en
language=${1:-$default_language}
sample_dataset_type=${2:-"vg"}
data_base=/mnt/petrelfs/songmingyang/songmingyang/data/mm
run_base=/mnt/petrelfs/songmingyang/songmingyang/runs/llava/preprocess
code_base=/mnt/petrelfs/songmingyang/code/mm/MAPO/m3apo/preprocess
cd $code_base

if [ $sample_dataset_type == "vg" ]
then
    image_folder=${data_base}/imgs/vg
    question_file=${data_base}/annotation/hadpo-data/hadpo/llava-v1.5/desc_data.json
   
    generation_dir_path=${run_base}/sft_on_palo_1000/desc_generations
    dataset_type=vg
else
    image_folder=${data_base}/imgs/coco/train2017
    question_file=${data_base}/annotation/LLaVA-Human-Preference-10K/llava_7b_v1_preference.json

    generation_dir_path=${run_base}/sft_on_palo_1000/human_generations
    dataset_type=human_preference_10k
fi

vg_path=${data_base}/annotation/vg

mkdir -p ${generation_dir_path}
generation_file=${generation_dir_path}/llava_sft_palo_generation_${dataset_type}_num${generation_num}_${language}.json

#--nodelist=SH-IDCA1404-10-140-54-[11,16]

gpus=1
cpus=16
quotatype="reserved"
OMP_NUM_THREADS=4 srun --partition=MoE --job-name="sampling" --mpi=pmi2 --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype} \
python /mnt/petrelfs/songmingyang/code/mm/MAPO/m3apo/preprocess/lvlm_sampling.py \
--model-path ${model_path} \
--question-file  ${question_file} \
--question_file_format "json" \
--image_folder ${image_folder} \
--answers-file  ${generation_file} \
--answers_file_format "json" \
--cd_alpha $cd_alpha \
--cd_beta $cd_beta \
--noise_step $noise_step \
--seed ${seed} \
--language ${language} \
--generation_num ${generation_num} \
--dataset_type ${dataset_type} \
--vg_path ${vg_path}