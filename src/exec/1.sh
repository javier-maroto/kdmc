for seed in 0
do
    for dataset in sbasic
    do
        for loss in std std_ml std_lnr
        do
            python src/main.py --dataset $dataset --loss $loss --n_epochs 100 --sch_gamma 0.95 --id $loss --batch_size 1024 --seed $seed
            for dataset_size in 100000 1000000
            do
                python src/main.py --dataset $dataset --loss $loss --n_epochs 100 --sch_gamma 0.95 --id $loss --batch_size 1024 --seed $seed --dataset_size $dataset_size
            done
        done
        loss=std_yml
        for alpha in 0.2 0.4 0.6 0.8
        do
            python src/main.py --dataset $dataset --loss $loss --n_epochs 100 --sch_gamma 0.95 --id $loss --batch_size 1024 --seed $seed --kt_alpha $alpha
            for dataset_size in 100000 1000000
            do
                python src/main.py --dataset $dataset --loss $loss --n_epochs 100 --sch_gamma 0.95 --id $loss --batch_size 1024 --seed $seed --dataset_size $dataset_size --kt_alpha $alpha
            done
        done
        for spr in 30 25 20
            do
            for loss in at at_ml at_lnr
            do
                python src/main.py --dataset $dataset --loss $loss --n_epochs 100 --sch_gamma 0.95 --id $loss --batch_size 1024 --seed $seed --atk pgd Linf $spr 0.25 7
                for dataset_size in 100000 1000000
                do
                    python src/main.py --dataset $dataset --loss $loss --n_epochs 100 --sch_gamma 0.95 --id $loss --batch_size 1024 --seed $seed --atk pgd Linf $spr 0.25 7 --dataset_size $dataset_size
                done
            done
            loss=at_yml
            for alpha in 0.2 0.4 0.6 0.8
            do
                python src/main.py --dataset $dataset --loss $loss --n_epochs 100 --sch_gamma 0.95 --id $loss --batch_size 1024 --seed $seed --atk pgd Linf $spr 0.25 7 --kt_alpha $alpha
                for dataset_size in 100000 1000000
                do
                    python src/main.py --dataset $dataset --loss $loss --n_epochs 100 --sch_gamma 0.95 --id $loss --batch_size 1024 --seed $seed --atk pgd Linf $spr 0.25 7 --dataset_size $dataset_size --kt_alpha $alpha
                done
            done
        done
    done
done