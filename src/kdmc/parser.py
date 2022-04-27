import argparse
import torch


def parse_args():
    # General parameters
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--id', help='Run id')
    parser.add_argument('--seed', default=0, type=int, help='Seed of the model / dataloader')
    parser.add_argument('--lr', default=0.1, type=float, help='Initial learning rate')
    parser.add_argument('--resume', '-r', type=str, help='resume from checkpoint')
    parser.add_argument('--arch', default='resnet', choices=['resnet'], help='Target model architecture')
    parser.add_argument('--sched', default='exp', choices=['1cyl', 'exp', 'fixed'], help='scheduler')
    parser.add_argument('--loss', help='loss used')
    parser.add_argument('--batch_size', default=256, type=int, help='Batch size')
    parser.add_argument('--n_epochs', default=50, type=int, help='Num of training epochs')
    parser.add_argument('--save_inter', action='store_true', help='save intermediate models (useful for kt intermediate models)')
    parser.add_argument('--save_freq', default=5, type=int, help='Save frequency (epochs)')
    parser.add_argument('--save_val', action='store_true', help='Skip test')
    parser.add_argument('--skip_test', action='store_true', help='Skip test')
    parser.add_argument('--root_path', default='.')
    # Parameters for adversarial training
    parser.add_argument('--atk', nargs="+", default=['pgd', 'Linf', '20', '0.25', '7'], type=str, help='Attack (name, **kwargs)')
    # Parameters for data
    parser.add_argument('--dataset', default='s1024', choices=['rml2016.10a', 's1024', 'sbasic', 'sbasic_nf'], help='Dataset used')
    parser.add_argument('--time_samples', type=int, help='Number of time samples of the IQ signal')
    parser.add_argument('--return_ml', action='store_true', help='Return the maximum likelihood')
    # Parameters for scheduler
    parser.add_argument('--sch_gamma', default=0.9, type=float, help='Gamma for exponential lr')
    # Parameters for AKD
    parser.add_argument('--kt_path', nargs="+", help='path of the stored models')
    parser.add_argument('--kt_alpha', type=float, help='Mixing rate (1.0 = only use kt preds)')
    parser.add_argument('--kt_beta', nargs="+", default=[], type=float, help='Mixing rate between the kt models (by default it mixes uniformly))')
    parser.add_argument('--ens_epoch', nargs="+", type=int, default=[], help='Epoch of the kt model (if intermediate saved). If None or -1, it uses the last one.')
    
    args =  parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    return args