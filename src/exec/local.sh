for seed in 0
do
    for dataset in sp0c20
    do
        for loss in at_ml at_lnr at_aml
        do
            python src/main.py --dataset $dataset --loss $loss --n_epochs 100 --sch_gamma 0.95 --id $loss --batch_size 1024 --seed $seed --data_path /mnt/Data/Datasets
        done
    done
done