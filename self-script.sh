#!/bin/sh

# for r in 0.0025 0.001 0.005 0.01
for r in 0.0025
do
    file='res/arxiv/'$r'_res.txt'
    # python train_self_supervised.py --dataset ogbn-arxiv --nlayers=2 --sgc=1 --lr_feat=0.01 --device=0  --lr_adj=0.01 --r=${r}  --seed=1 --inner=3  --epochs=1000  --save=1 >> $file
    python train_self_supervised.py --dataset ogbn-arxiv --nlayers=2 --sgc=1 --lr_feat=0.01 --device=0  --lr_adj=0.01 --reduction_rate=${r}  --seed=1  --epochs=50  --save=1 >> $file
done