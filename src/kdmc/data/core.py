import numpy as np
from torch.utils.data import DataLoader, Subset

from kdmc.data.rml2016_10a import get_rml2016_10a_datasets
from kdmc.data.s1024 import get_s1024_datasets
from kdmc.data.sbasic_nf import get_sbasic_datasets
import kdmc.data.synthetic as ds


def get_datasets(args):
    if args.dataset == 'rml2016.10a':
        return get_rml2016_10a_datasets(args.data_path)
    elif args.dataset == 's1024':
        return get_s1024_datasets(args.data_path)
    elif args.dataset == 'sbasic_nf':
        return get_sbasic_datasets(args.data_path, args.time_samples, args.seed, use_filters=False)
    else:
        if args.dataset == 'sbasic':
            dataset = ds.SAWGNp0c20(args.data_path, args.time_samples, args.dataset_size)
        elif args.dataset == 'sawgn':
            dataset = ds.SAWGNp1c20(args.data_path, args.time_samples, args.dataset_size)
        elif args.dataset == 'sawgn2p':
            dataset = ds.SAWGNp2c20(args.data_path, args.time_samples, args.dataset_size)
        elif args.dataset == 'srml2016.10a':
            dataset = ds.SRML2016_10A(args.data_path, args.dataset_size)
        elif args.dataset == 'srml2018':
            dataset = ds.SRML2018(args.data_path, args.dataset_size)
        elif args.dataset == 'sp0c20':
            dataset = ds.Sp0c20(args.data_path, args.time_samples, args.dataset_size)
        else:
            raise NotImplementedError(f"dataset not implemented: {args.dataset}")
        return ds.split_synthetic_dataset(dataset, args.seed)
    

def get_num_classes(dataset):
    return len(get_classes(dataset))


def get_classes(dataset):
    if dataset in ('rml2016.10a', 'srml2016.10a'):
        return (
            "B-FM",
            "DSB-AM",
            "SSB-AM",
            "CPFSK",
            "GFSK",
            "PAM4",
            "BPSK",
            "QPSK",
            "8-PSK",
            "16-QAM",
            "64-QAM"
        )
    elif dataset in ('s1024', 'sbasic', 'sbasic_nf', 'sawgn', 'sawgn2p', 'sp0c20'):
        return (
            "BPSK", "QPSK", "8-PSK",
            "16-APSK", "32-APSK", "64-APSK", "128-APSK", "256-APSK",
            "PAM4", "16-QAM", "32-QAM", "64-QAM", "128-QAM", "256-QAM",
            'GFSK', 'CPFSK', 'B-FM', 'DSB-AM', 'SSB-AM', 'OQPSK'
        )
    elif dataset == 'srml2018':
        return ds.RML2018.classes
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
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers, pin_memory=True, persistent_workers=True)
    testloader = DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers, pin_memory=True, persistent_workers=True)
    return trainloader, testloader