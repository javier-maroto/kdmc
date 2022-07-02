import argparse
import torch
from pathlib import Path


def parse_args(args=None):
    # General parameters
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--id', type=str, help='Run id', default='test')
    parser.add_argument('--seed', default=0, type=int, help='Seed of the model / dataloader')
    parser.add_argument('--resume', '-r', type=str, help='resume from checkpoint')
    parser.add_argument('--arch', default='resnet', choices=['resnet'], help='Target model architecture')
    parser.add_argument('--profile', action='store_true', help='Profile the model')
    parser.add_argument('--saved_model', type=str, help='Path to model to load')
    # Path parameters
    parser.add_argument('--root_path', default='.')
    parser.add_argument('--data_path')
    # Parameters for training
    parser.add_argument('--loss', default='std', help='loss used')
    parser.add_argument('--n_epochs', default=50, type=int, help='Num of training epochs')
    # Parameters for optimizer
    parser.add_argument('--lr', default=0.01, type=float, help='Initial learning rate')
    parser.add_argument('--grad_clip', default=5, type=float, help='Gradient clipping')
    # Parameters for scheduler
    parser.add_argument('--sched', default='exp', choices=['1cyl', 'exp', 'fixed'], help='scheduler')
    parser.add_argument('--sch_gamma', default=0.95, type=float, help='Gamma for exponential lr')
    # Parameters for validation
    parser.add_argument('--save_inter', action='store_true', help='save intermediate models (useful for kt intermediate models)')
    parser.add_argument('--save_freq', default=5, type=int, help='Save frequency (epochs)')
    parser.add_argument('--save_val', action='store_true', help='Skip test')
    parser.add_argument('--skip_test', action='store_true', help='Skip test')
    # Parameters for adversarial training
    parser.add_argument('--atk', nargs="+", default=['pgd', 'Linf', '20', '0.25', '7'], type=str, help='Attack (name, **kwargs)')
    # Parameters for data
    parser.add_argument('--dataset', default='s1024', choices=['rml2016.10a', 's1024', 'sbasic', 'sbasic_nf', 'sawgn', 'sawgn2p', 'srml2016.10a', 'sp0c20'], help='Dataset used')
    parser.add_argument('--dataset_size', type=int, help='Dataset size')
    parser.add_argument('--batch_size', default=256, type=int, help='Batch size')
    parser.add_argument('--n_batches', default=-1, type=int, help='Number of batches to use')
    parser.add_argument('--n_workers', default=2, type=int, help='Number of dataloader workers')
    parser.add_argument('--time_samples', type=int, help='Number of time samples of the IQ signal')
    parser.add_argument('--return_ml', action='store_true', help='Return the maximum likelihood')  # DEPRECATED
    # Parameters for AKD and self distillation
    parser.add_argument('--kt_path', nargs="+", help='path of the stored models')
    parser.add_argument('--kt_alpha', type=float, help='Mixing rate (1.0 = only use kt preds)')
    parser.add_argument('--kt_beta', nargs="+", default=[], type=float, help='Mixing rate between the kt models (by default it mixes uniformly))')
    parser.add_argument('--ens_epoch', nargs="+", type=int, default=[], help='Epoch of the kt model (if intermediate saved). If None or -1, it uses the last one.')
    
    args =  parser.parse_args(args)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.root_path = Path(args.root_path)
    # If None, data is in the same root folder
    if args.data_path is None:
        args.data_path = args.root_path / 'data'
    else:
        args.data_path = Path(args.data_path)
    # Check only 10 batches when profiling
    if args.profile:
        args.n_batches = 10
        args.n_epochs = 1

    return args