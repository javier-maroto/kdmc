python src/main.py --dataset rml2016.10a --loss std --n_epochs 10 --id std --data_path D:/Datasets

python src/main.py --dataset s1024 --loss std --n_epochs 5

python src/main.py --dataset sbasic --loss std --n_epochs 100 --sch_gamma 0.95 --id std --data_path D:/Datasets
python src/main.py --dataset sbasic --loss std --n_epochs 100 --sch_gamma 0.95 --id std_ml --return_ml


python src/main.py --dataset sp0c20 --loss std --n_epochs 100 --sch_gamma 0.95 --id std --data_path D:/Datasets --dataset_size 1000000 --batch_size 1024
python src/main.py --dataset sp0c20 --loss std_ml --n_epochs 100 --sch_gamma 0.95 --id std_ml --data_path D:/Datasets --dataset_size 1000000 --batch_size 1024
python src/main.py --dataset sp0c20 --loss std_sml --n_epochs 100 --sch_gamma 0.95 --id std_sml75 --data_path D:/Datasets --dataset_size 1000000 --batch_size 1024 --kt_alpha 0.75

python src/main.py --dataset srml2016.10a --loss std --n_epochs 50 --data_path D:/Datasets --id std --dataset_size 1000000

python src/main.py --dataset sbasic_nf --loss std --n_epochs 100 --sch_gamma 0.95 --id std --time_samples 128 
python src/main.py --dataset sbasic_nf --loss std --n_epochs 100 --sch_gamma 0.95 --id std_ml --time_samples 128 --return_ml

python src/main.py --dataset sbasic_nf --loss at --n_epochs 100 --sch_gamma 0.95 --id at --time_samples 128
python src/main.py --dataset sbasic_nf --loss at --n_epochs 100 --sch_gamma 0.95 --id at_ml --time_samples 128 --return_ml

# Run synthetic RML2018 dataset
python src/main.py --dataset srml2018 --loss std --n_epochs 100 --sch_gamma 0.95 --id std --batch_size 1024 --seed 0 --data_path /media/javier/Data/Datasets 


export CUDA_VISIBLE_DEVICES="0"
python src/main.py --dataset sp0c20 --loss std_ml --n_epochs 100 --sch_gamma 0.95 --id std_ml --batch_size 1024
python src/main.py --dataset sp0c20 --loss at --n_epochs 100 --sch_gamma 0.95 --id at --batch_size 1024
python src/main.py --dataset sp0c20 --loss at_ml --n_epochs 100 --sch_gamma 0.95 --id at_ml --batch_size 1024
