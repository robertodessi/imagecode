#!/bin/bash 
date

GPUID=0
for n_tokens in 1 2 10 20
do
    for prompt_init in "random" "random_embedding"
    do
        output="output/run_tokens"$n_tokens"_init_"$prompt_init
        CUDA_VISIBLE_DEVICES=$GPUID python contra_soft_prompt.py --n_tokens $n_tokens --prompt_init $prompt_init > $output".out" 2> $output".err" &
        echo " $GPUID "
        GPUID=$[$GPUID +1]
    done
done

date
echo DONE
