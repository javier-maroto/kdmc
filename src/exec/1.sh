for seed in {0..3}
do
python src/main.py --dataset sp0c20 --loss std --n_epochs 100 --sch_gamma 0.95 --id std --batch_size 1024 --seed $seed
python src/main.py --dataset sp0c20 --loss std_ml --n_epochs 100 --sch_gamma 0.95 --id std_ml --batch_size 1024 --seed $seed
python src/main.py --dataset sp0c20 --loss std_lnr --n_epochs 100 --sch_gamma 0.95 --id std_lnr --batch_size 1024 --seed $seed
python src/main.py --dataset sbasic --loss std --n_epochs 100 --sch_gamma 0.95 --id std --batch_size 1024 --seed $seed
python src/main.py --dataset sbasic --loss std_ml --n_epochs 100 --sch_gamma 0.95 --id std_ml --batch_size 1024 --seed $seed
python src/main.py --dataset sbasic --loss std_lnr --n_epochs 100 --sch_gamma 0.95 --id std_lnr --batch_size 1024 --seed $seed
python src/main.py --dataset sawgn2p --loss std --n_epochs 100 --sch_gamma 0.95 --id std --batch_size 1024 --seed $seed
python src/main.py --dataset sawgn2p --loss std_ml --n_epochs 100 --sch_gamma 0.95 --id std_ml --batch_size 1024 --seed $seed
python src/main.py --dataset sawgn2p --loss std_lnr --n_epochs 100 --sch_gamma 0.95 --id std_lnr --batch_size 1024 --seed $seed

python src/main.py --dataset sp0c20 --loss at --n_epochs 100 --sch_gamma 0.95 --id at --batch_size 1024 --seed $seed
python src/main.py --dataset sp0c20 --loss at_ml --n_epochs 100 --sch_gamma 0.95 --id at_ml --batch_size 1024 --seed $seed
python src/main.py --dataset sbasic --loss at --n_epochs 100 --sch_gamma 0.95 --id at --batch_size 1024 --seed $seed
python src/main.py --dataset sbasic --loss at_ml --n_epochs 100 --sch_gamma 0.95 --id at_ml --batch_size 1024 --seed $seed
python src/main.py --dataset sawgn2p --loss at --n_epochs 100 --sch_gamma 0.95 --id at --batch_size 1024 --seed $seed
python src/main.py --dataset sawgn2p --loss at_ml --n_epochs 100 --sch_gamma 0.95 --id at_ml --batch_size 1024 --seed $seed
for dataset_size in 100000, 1000000
do
python src/main.py --dataset sp0c20 --loss std --n_epochs 100 --sch_gamma 0.95 --id std --batch_size 1024 --dataset_size $dataset_size --seed $seed
python src/main.py --dataset sp0c20 --loss std_ml --n_epochs 100 --sch_gamma 0.95 --id std_ml --batch_size 1024 --dataset_size $dataset_size --seed $seed
python src/main.py --dataset sp0c20 --loss std_lnr --n_epochs 100 --sch_gamma 0.95 --id std_lnr --batch_size 1024 --dataset_size $dataset_size --seed $seed
python src/main.py --dataset sbasic --loss std --n_epochs 100 --sch_gamma 0.95 --id std --batch_size 1024 --dataset_size $dataset_size --seed $seed
python src/main.py --dataset sbasic --loss std_ml --n_epochs 100 --sch_gamma 0.95 --id std_ml --batch_size 1024 --dataset_size $dataset_size --seed $seed
python src/main.py --dataset sbasic --loss std_lnr --n_epochs 100 --sch_gamma 0.95 --id std_lnr --batch_size 1024 --dataset_size $dataset_size --seed $seed
python src/main.py --dataset sawgn2p --loss std --n_epochs 100 --sch_gamma 0.95 --id std --batch_size 1024 --dataset_size $dataset_size --seed $seed
python src/main.py --dataset sawgn2p --loss std_ml --n_epochs 100 --sch_gamma 0.95 --id std_ml --batch_size 1024 --dataset_size $dataset_size --seed $seed
python src/main.py --dataset sawgn2p --loss std_lnr --n_epochs 100 --sch_gamma 0.95 --id std_lnr --batch_size 1024 --dataset_size $dataset_size --seed $seed

python src/main.py --dataset sp0c20 --loss at --n_epochs 100 --sch_gamma 0.95 --id at --batch_size 1024 --dataset_size $dataset_size --seed $seed
python src/main.py --dataset sp0c20 --loss at_ml --n_epochs 100 --sch_gamma 0.95 --id at_ml --batch_size 1024 --dataset_size $dataset_size --seed $seed
python src/main.py --dataset sbasic --loss at --n_epochs 100 --sch_gamma 0.95 --id at --batch_size 1024 --dataset_size $dataset_size --seed $seed
python src/main.py --dataset sbasic --loss at_ml --n_epochs 100 --sch_gamma 0.95 --id at_ml --batch_size 1024 --dataset_size $dataset_size --seed $seed
python src/main.py --dataset sawgn2p --loss at --n_epochs 100 --sch_gamma 0.95 --id at --batch_size 1024 --dataset_size $dataset_size --seed $seed
python src/main.py --dataset sawgn2p --loss at_ml --n_epochs 100 --sch_gamma 0.95 --id at_ml --batch_size 1024 --dataset_size $dataset_size --seed $seed
done
done