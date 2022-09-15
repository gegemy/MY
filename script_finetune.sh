#!/bin/sh

for r in 0.0025 0.001 0.005 0.01
# for r in 0.001 
do
    # for train_iters in 600 100 50 10
    for train_iters in 600
    # for ((train_iters=1; train_iters<=600; train_iters++));
    do
        file='res/arxiv/'$r'_trainset_finetune_init=False_iters_res.txt'
        # echo $train_iters >> $file
        python fine_tune.py --dataset ogbn-arxiv --nlayers=2 --sgc=1 --lr_feat=0.01 --gpu_id=0  --lr_adj=0.01 --train_iters=${train_iters} --r=${r}  --seed=1 >> $file
    done
done