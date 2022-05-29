import random
import numpy as np
import torch
import wandb
import torch.profiler as tpf

from kdmc.data.core import create_dataloaders, get_datasets
from kdmc.parser import parse_args
from kdmc.train.core import create_model, create_scheduler, get_trainer


def main():
    # Detect when NaN appears
    torch.autograd.set_detect_anomaly(True)

    args = parse_args()

    wandb.init(project=f"kdmc_{args.dataset}", name=args.id, dir=args.root_path)
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
    net = create_model(args)

    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr,
                        momentum=0.9, weight_decay=5e-4)
    scheduler, schd_updt = create_scheduler(optimizer, args, args.n_epochs, len(trainloader))

    start_epoch = 1

    trainer = get_trainer(args, net, trainloader, testloader, optimizer, scheduler, schd_updt, args.save_freq)
    if args.profile:
        with tpf.profile(
                activities=[tpf.ProfilerActivity.CPU, tpf.ProfilerActivity.CUDA],
                on_trace_ready=tpf.tensorboard_trace_handler(dir_name=args.root_path.joinpath('profiler')),
                record_shapes=True,  # record shapes of operator inputs
                profile_memory=True,  # record tensor memory allocation
            ) as prof:
            trainer.train(start_epoch, profiler=prof)
        return
    for epoch in range(start_epoch, args.n_epochs + 1):
        trainer.loop(epoch)
    if not args.skip_test:
        trainer.test()


if __name__ == "__main__":
    main()
