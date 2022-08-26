#python src/main_test.py --dataset rml2018r --saved_model rml2018r/resnet/std/0/ckpt_last.pth --id std-r-r --debug
#python src/main_test.py --dataset rml2018r --saved_model srml2018/resnet/std/0/ckpt_last.pth --id std-s-r
#python src/main_test.py --dataset rml2018r --saved_model srml2018/resnet/std_ml/0/ckpt_last.pth --id std_ml-s-r

#python src/main_test.py --dataset srml2018 --saved_model srml2018/resnet/std/0/ckpt_last.pth --id std-s-s --debug
#python src/main_test.py --dataset srml2018 --saved_model rml2018r/resnet/std/0/ckpt_last.pth --id std-r-s --debug

for seed in {0..3}
do
#python src/main.py --dataset srml2018 --loss at --n_epochs 100 --sch_gamma 0.95 --id at --batch_size 1024 --seed $seed
python src/main.py --dataset srml2018 --loss at_ml --n_epochs 100 --sch_gamma 0.95 --id at_ml --batch_size 1024 --seed $seed
done