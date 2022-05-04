python src/main.py --dataset rml2016.10a --loss std --n_epochs 5

python src/main.py --dataset s1024 --loss std --n_epochs 5

python src/main.py --dataset sbasic --loss std --n_epochs 100 --sch_gamma 0.95 --id std
python src/main.py --dataset sbasic --loss std --n_epochs 100 --sch_gamma 0.95 --id std_ml --return_ml


python src/main.py --dataset sbasic_nf --loss std --n_epochs 100 --sch_gamma 0.95 --id std --time_samples 128
python src/main.py --dataset sbasic_nf --loss std --n_epochs 100 --sch_gamma 0.95 --id std_ml --time_samples 128 --return_ml

python src/main.py --dataset sbasic_nf --loss at --n_epochs 100 --sch_gamma 0.95 --id at --time_samples 128
python src/main.py --dataset sbasic_nf --loss at --n_epochs 100 --sch_gamma 0.95 --id at_ml --time_samples 128 --return_ml