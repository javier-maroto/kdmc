python src/main_test.py --dataset sawgn --saved_model sbasic/resnet/std/0/ckpt_last.pth --id std-b-a
python src/main_test.py --dataset sawgn2p --saved_model sbasic/resnet/std/0/ckpt_last.pth --id std-b-a2
python src/main_test.py --dataset sawgn --saved_model sbasic/resnet/std_ml/0/ckpt_last.pth --id std_ml-b-a
python src/main_test.py --dataset sawgn2p --saved_model sbasic/resnet/std_ml/0/ckpt_last.pth --id std_ml-b-a2

python src/main_test.py --dataset sbasic --saved_model sawgn/resnet/std/0/ckpt_last.pth --id std-a-b
python src/main_test.py --dataset sawgn2p --saved_model sawgn/resnet/std/0/ckpt_last.pth --id std-a-a2
python src/main_test.py --dataset sbasic --saved_model sawgn/resnet/std_ml/0/ckpt_last.pth --id std_ml-a-b
python src/main_test.py --dataset sawgn2p --saved_model sawgn/resnet/std_ml/0/ckpt_last.pth --id std_ml-a-a2

python src/main_test.py --dataset sbasic --saved_model sawgn2p/resnet/std/0/ckpt_last.pth --id std-a2-b
python src/main_test.py --dataset sawgn --saved_model sawgn2p/resnet/std/0/ckpt_last.pth --id std-a2-a
python src/main_test.py --dataset sbasic --saved_model sawgn2p/resnet/std_ml/0/ckpt_last.pth --id std_ml-a2-b
python src/main_test.py --dataset sawgn --saved_model sawgn2p/resnet/std_ml/0/ckpt_last.pth --id std_ml-a2-a
python src/main_test.py --dataset sbasic --saved_model sawgn2p/resnet/std_hd90/0/ckpt_last.pth --id std_hd90-a2-b
python src/main_test.py --dataset sawgn --saved_model sawgn2p/resnet/std_hd90/0/ckpt_last.pth --id std_hd90-a2-a