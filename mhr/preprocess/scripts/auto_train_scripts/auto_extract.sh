language_list=(en ar  bg  bn  de  el  es  fa  fr  gu  hi  id  it  ja  jv  ko  ml  mr  ms  my  nl  pt  ru  sv  sw  ta  te  th  tr  uk  ur  vi  zh)
strategis=(language preference hallucination)

code_base=/mnt/petrelfs/songmingyang/code/mm/MAPO/mhr/preprocess
scripts_path=${code_base}/scripts
log_path=${code_base}/scripts/auto_train_scripts/logs/extract
mkdir -p ${log_path}
cd $scripts_path


for strategy in ${strategis[@]}
do
    for language in ${language_list[@]}
    do
        echo "Sampling for ${strategy} in ${language}"
        bash /mnt/petrelfs/songmingyang/code/mm/MAPO/mhr/preprocess/scripts/srun_extract_dpo_data.sh ${language} ${strategy} 1>${log_path}/extract_${strategy}_${language}.log 2>&1 &
        sleep 1
    done
done