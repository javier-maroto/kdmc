for seed in 3
do
    for dataset in sp0c20
    do
        for loss in at_lnr at_ml
        do
            python src/main.py --dataset $dataset --loss $loss --n_epochs 100 --sch_gamma 0.95 --id $loss --batch_size 1024 --seed $seed
        done
    done
done