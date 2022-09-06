for seed in {0..3}
do
    for loss in std std_ml std_lnr at at_ml at_aml
    do
        for dataset in sbasic sawgn2p sp0c20
        do
            for spr in 30 25 20
            do
                python src/main.py --dataset $dataset --loss $loss --n_epochs 100 --sch_gamma 0.95 --id $loss --batch_size 1024 --seed $seed --atk pgd Linf $spr 0.25 7
                for dataset_size in 100000 1000000
                do
                    python src/main.py --dataset $dataset --loss $loss --n_epochs 100 --sch_gamma 0.95 --id $loss --batch_size 1024 --seed $seed --atk pgd Linf $spr 0.25 7 --dataset_size $dataset_size
                done
            done
        done
    done
done