import random
import numpy as np
import torch
import wandb
from kdmc.data.core import create_dataloaders, get_datasets

from kdmc.parser import parse_args
from kdmc.train.core import create_model, get_trainer


def main():
    
    args = parse_args()

    wandb.init(project=f"kdmc_{args.dataset}", name=args.id)
    wandb.config.update(args)

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Data
    print('==> Preparing data..')
    trainset, testset = get_datasets(args)
    trainloader, testloader = create_dataloaders(args, trainset, testset)
    args.time_samples = trainset.dataset.time_samples

    # Model
    print('==> Building model..')
    net = None

    trainer = get_trainer(args, net, trainloader, testloader, None, None, None, args.save_freq)
    trainer.test()


if __name__ == "__main__":
    main()
