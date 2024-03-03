begin_id=2567290
end_id=2567347

for i in $(seq $begin_id $end_id)
do
   scancel $i
done
