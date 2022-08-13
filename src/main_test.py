from datetime import datetime
import os
import random
import numpy as np
import torch
import wandb
from kdmc.data.core import create_dataloaders, get_classes, get_datasets

from kdmc.parser import parse_args
from kdmc.test import FullTester
from kdmc.train.core import create_model

import logging


def main():
    
    args = parse_args()

    wandb.init(project=f"kdmc_{args.dataset}", name=args.id)
    wandb.config.update(args)

    os.makedirs('logs', exist_ok=True)
    loglevel = logging.DEBUG if args.debug else logging.INFO
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logging.basicConfig(level=loglevel, format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', filename=f'logs/{now}_{args.id}.log')

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Data
    logging.info('==> Preparing data..')
    trainset, testset = get_datasets(args)
    _, testloader = create_dataloaders(args, trainset, testset)
    args.time_samples = trainset.dataset.time_samples

    # Model
    print('==> Building model..')
    net = create_model(args)
    if args.saved_model is not None:
        ckpt = torch.load(f"{args.root_path}/checkpoint/{args.saved_model}")
    else:
        ckpt = torch.load(f"{args.root_path}/checkpoint/{args.dataset}/{args.arch}/{args.id}/{args.seed}/ckpt_last.pth")
    net.load_state_dict(ckpt["net"])
    net.to(args.device)
    net.eval()

    classes = get_classes(args.dataset)
    tester = FullTester(args, net, classes)
    tester.test(testloader)


if __name__ == "__main__":
    main()
