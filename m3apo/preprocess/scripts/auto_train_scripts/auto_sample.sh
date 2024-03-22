language_list=(en de sv fr ar zh ja ko bn sw mr ru ur bg th hi uk) 
datasets=(vg human)

code_base=/mnt/petrelfs/songmingyang/code/mm/MAPO/m3apo/preprocess
scripts_path=${code_base}/scripts
log_path=${code_base}/scripts/auto_train_scripts/logs
mkdir -p ${log_path}
cd $scripts_path

for dataset in ${datasets[@]}
do
    for language in ${language_list[@]}
    do
        echo "Sampling for ${dataset} in ${language}"
        bash /mnt/petrelfs/songmingyang/code/mm/MAPO/m3apo/preprocess/scripts/srun_lvlm_sample.sh ${language} ${dataset} 1>${log_path}/auto_sample_${dataset}_${language}.log 2>&1 &
    done
done