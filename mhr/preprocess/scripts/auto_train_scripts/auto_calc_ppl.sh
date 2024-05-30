language_list=(en ar  bg  bn  de  el  es  fa  fr  gu  hi  id  it  ja  jv  ko  ml  mr  ms  my  nl  pt  ru  sv  sw  ta  te  th  tr  uk  ur  vi  zh)
alignment_strategy=(language hallucination preference)

code_base=/mnt/petrelfs/songmingyang/code/mm/MAPO/m3apo/preprocess
scripts_path=${code_base}/scripts
log_path=${code_base}/scripts/auto_train_scripts/logs/bleu
mkdir -p ${log_path}
cd $scripts_path

for srategy in ${alignment_strategy[@]}
do
    for language in ${language_list[@]}
    do
        echo "Sampling for ${srategy} in ${language}"
        bash /mnt/petrelfs/songmingyang/code/mm/MAPO/m3apo/preprocess/scripts/srun_desc_calc_bleu.sh ${language} ${srategy} 1>${log_path}/calc_bleu_${srategy}_${language}.log 2>&1 &
        sleep 1
    done
done