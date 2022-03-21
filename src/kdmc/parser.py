import argparse
import torch


def parse_args():
    # General parameters
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--id', help='Run id')
    parser.add_argument('--seed', default=0, type=int, help='Seed of the model / dataloader')
    parser.add_argument('--lr', default=0.1, type=float, help='Initial learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--arch', default='resnet', choices=['resnet'], help='Target model architecture')
    parser.add_argument('--sched', default='1cyl', choices=['1cyl', 'exp', 'fixed'], help='scheduler')
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
    parser.add_argument('--dataset', default='cifar10', choices=['rml2016.10a'], help='Dataset used')
    parser.add_argument('--time_samples', type=int, help='Number of time samples of the IQ signal')
    
    args =  parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    return args