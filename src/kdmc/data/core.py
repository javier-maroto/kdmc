import numpy as np
from torch.utils.data import DataLoader, Subset

from kdmc.data.rml2016_10a import get_rml2016_10a_datasets
from kdmc.data.s1024 import get_s1024_datasets
from kdmc.data.sbasic import get_sbasic_datasets
from kdmc.data.sawgn import get_sawgn_datasets


def get_datasets(args):
    if args.dataset == 'rml2016.10a':
        return get_rml2016_10a_datasets(args.data_path)
    elif args.dataset == 's1024':
        return get_s1024_datasets(args.data_path)
    elif args.dataset == 'sbasic':
        return get_sbasic_datasets(args.data_path, args.time_samples, args.seed)
    elif args.dataset == 'sbasic_nf':
        return get_sbasic_datasets(args.data_path, args.time_samples, args.seed, use_filters=False)
    elif args.dataset == 'sawgn':
        return get_sawgn_datasets(args.data_path, args.time_samples, args.seed)
    else:
        raise NotImplementedError(f"dataset not implemented: {args.dataset}")


def get_num_classes(dataset):
    if dataset == 'rml2016.10a':
        return 11
    elif dataset in ('s1024', 'sbasic', 'sbasic_nf', 'sawgn'):
        return 20
    else:
        raise NotImplementedError(f"dataset not implemented: {dataset}")


def get_input_dims(args):
    pass


def create_dataloaders(args, trainset, testset):
    if args.n_batches != -1:
        if args.n_batches * args.batch_size > len(trainset):
            raise ValueError(f"n_batches * batch_size > len(trainset)")
        trainset = Subset(trainset, np.random.choice(len(trainset), args.n_batches * args.batch_size, replace=False))
    trainloader = DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers, pin_memory=True)
    testloader = DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers, pin_memory=True)
    return trainloader, testloader