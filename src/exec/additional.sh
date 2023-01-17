# Experiments used in the paper
gpu=3
for seed in {0..3}
do
    for arch in cnn cldnn
    do
        for dataset in sbasic sawgn2p sp0c20
        do
            for loss in std std_lnr std_ml at at_lnr at_aml
            do
                pueue add -g 1 -- CUDA_VISIBLE_DEVICES=${gpu} python src/main.py --arch ${arch} --dataset ${dataset} --loss ${loss} --n_epochs 100 --sch_gamma 0.95 --id ${loss} --batch_size 1024 --seed ${seed} --data_path data
            done
        done
    done
done
