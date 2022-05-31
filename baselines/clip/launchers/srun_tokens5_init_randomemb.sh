#!/bin/bash

date
echo "BEGIN"

cd ../
python /private/home/rdessi/imagecode/baselines/clip/contra_soft_prompt.py --n_tokens 5 --prompt_init "random_embedding"

date
echo "ALL DONE"
