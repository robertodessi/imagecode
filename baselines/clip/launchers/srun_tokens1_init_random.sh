#!/bin/bash

date
echo "BEGIN"

cd ../
python ~/imagecode/baselines/clip/contra_soft_prompt.py --n_tokens 1 --prompt_init "random"

date
echo "ALL DONE"
