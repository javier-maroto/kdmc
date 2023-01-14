# Experiments used in the paper
for seed in 0..3
do
    for arch in cldnn cnn
    do
        for dataset in sbasic sawgn2p sp0c20
        do
            for loss in std std_lnr std_ml at at_lnr at_aml
            do
                pueue add -g 1 -- 'CUDA_VISIBLE_DEVICES=$PUEUE_WORKER_ID python src/main.py --arch $arch --dataset $dataset --loss $loss --n_epochs 100 --sch_gamma 0.95 --id $loss --batch_size 1024 --seed $seed --data_path data'
            done
        done
    done
done


pueue add -g 1 -- 'CUDA_VISIBLE_DEVICES=$PUEUE_WORKER_ID python src/main.py --arch cldnn --dataset sbasic --loss std --n_epochs 100 --sch_gamma 0.95 --id std --batch_size 1024 --seed 0 --data_path data'