begin_id=2567356
end_id=2567409

for i in $(seq $begin_id $end_id)
do
   scancel $i
done
