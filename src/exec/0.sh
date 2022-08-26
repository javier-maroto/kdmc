python src/main.py --dataset sp0c20 --loss std --n_epochs 100 --sch_gamma 0.95 --id std --batch_size 1024 --dataset_size 1000000
python src/main.py --dataset sp0c20 --loss std_ml --n_epochs 100 --sch_gamma 0.95 --id std_ml --batch_size 1024 --dataset_size 1000000
python src/main.py --dataset sp0c20 --loss std_lnr --n_epochs 100 --sch_gamma 0.95 --id std_lnr --batch_size 1024 --dataset_size 1000000

python src/main.py --dataset sbasic --loss std --n_epochs 100 --sch_gamma 0.95 --id std --batch_size 1024 --dataset_size 1000000
python src/main.py --dataset sbasic --loss std_ml --n_epochs 100 --sch_gamma 0.95 --id std_ml --batch_size 1024 --dataset_size 1000000
python src/main.py --dataset sbasic --loss std_lnr --n_epochs 100 --sch_gamma 0.95 --id std_lnr --batch_size 1024 --dataset_size 1000000

python src/main.py --dataset sawgn2p --loss std --n_epochs 100 --sch_gamma 0.95 --id std --batch_size 1024 --dataset_size 1000000
python src/main.py --dataset sawgn2p --loss std_ml --n_epochs 100 --sch_gamma 0.95 --id std_ml --batch_size 1024 --dataset_size 1000000
python src/main.py --dataset sawgn2p --loss std_lnr --n_epochs 100 --sch_gamma 0.95 --id std_lnr --batch_size 1024 --dataset_size 1000000