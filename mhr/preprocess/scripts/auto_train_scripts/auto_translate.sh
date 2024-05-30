language_list=(ar  bg  bn  de  el  es  fa  fr  gu  hi  id  it  ja  jv  ko  ml  mr  ms  my  nl  pt  ru  sv  sw  ta  te  th  tr  uk  ur  vi  zh)

for language in "${language_list[@]}"; do
    bash /mnt/petrelfs/songmingyang/code/mm/MAPO/mhr/preprocess/scripts/srun_desc_calc_bleu_translate.sh $language 1>/mnt/petrelfs/songmingyang/code/mm/MAPO/mhr/preprocess/scripts/auto_train_scripts/logs/trans_log_${language}.txt 2>&1 &
    echo "translating $language"
    sleep 1
done