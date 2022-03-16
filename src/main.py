import random
import numpy as np
import torch
import wandb
from kdmc.attack.core import parse_attack
from kdmc.data.core import create_dataloaders, get_datasets

from kdmc.parser import parse_args
from kdmc.train.core import create_model, create_scheduler, get_trainer


def main():
    
    args = parse_args()

    wandb.init(project=f"dk_{args.dataset}", name=args.id)
    wandb.config.update(args)

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Data
    print('==> Preparing data..')
    trainset, testset = get_datasets(args)
    trainloader, testloader = create_dataloaders(args, trainset, testset)

    # Model
    print('==> Building model..')
    net = create_model(args)

    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr,
                        momentum=0.9, weight_decay=5e-4)
    scheduler, schd_updt = create_scheduler(optimizer, args, args.n_epochs, len(trainloader))

    start_epoch = 1
    if args.resume:
        raise NotImplementedError

    attack = parse_attack(net, args.atk)
    trainer = get_trainer(args, net, trainloader, testloader, optimizer, scheduler, schd_updt, args.save_freq)
    for epoch in range(start_epoch, args.n_epochs + 1):
        trainer.loop(epoch)
    if not args.skip_test:
        trainer.test()


if __name__ == "__main__":
    main()
