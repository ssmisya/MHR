language_list=(en ar  bg  bn  de  el  es  fa  fr  gu  hi  id  it  ja  jv  ko  ml  mr  ms  my  nl  pt  ru  sv  sw  ta  te  th  tr  uk  ur  vi  zh)
datasets=(vg)

code_base=/mnt/petrelfs/songmingyang/code/mm/MAPO/mhr/preprocess
scripts_path=${code_base}/scripts
log_path=${code_base}/scripts/auto_train_scripts/logs
mkdir -p ${log_path}
cd $scripts_path

for dataset in ${datasets[@]}
do
    for language in ${language_list[@]}
    do
        echo "Sampling for ${dataset} in ${language}"
        bash /mnt/petrelfs/songmingyang/code/mm/MAPO/mhr/preprocess/scripts/srun_lvlm_sample.sh ${language} ${dataset} 1>${log_path}/sample_${dataset}_${language}.log 2>&1 &
        sleep 1
    done
done