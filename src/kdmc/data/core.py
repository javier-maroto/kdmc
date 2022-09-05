import numpy as np
from torch.utils.data import DataLoader, RandomSampler

from kdmc.data.rml2016_10a import get_rml2016_10a_datasets
from kdmc.data.s1024 import get_s1024_datasets
from kdmc.data.sbasic_nf import get_sbasic_datasets
import kdmc.data.synthetic as ds
from kdmc.data.rml2018 import get_rml2018_datasets, RML2018_D
from kdmc.data.rml2018r import get_rml2018r_datasets
from .utils import BatchMixerSampler, DatasetMixer, SubsetDataset


def get_datasets(args):
    if args.dataset == 'rml2016.10a':
        train, test = get_rml2016_10a_datasets(args.data_path)
    elif args.dataset == 'rml2018':
        train, test = get_rml2018_datasets(args.data_path)
    elif args.dataset == 'rml2018r':
        train, test = get_rml2018r_datasets(args.data_path)
    elif args.dataset == 's1024':
        train, test = get_s1024_datasets(args.data_path)
    elif args.dataset == 'sbasic_nf':
        train, test = get_sbasic_datasets(args.data_path, args.time_samples, args.seed, use_filters=False)
    elif args.dataset == 'sm_rml2018':  # Synthetic mix
        real_train, test = get_rml2018r_datasets(args.data_path)
        synth_dataset = ds.SRML2018(args.data_path, args.dataset_size)
        synth_train, _ = ds.split_synthetic_dataset(synth_dataset, args.seed)
        train = DatasetMixer([real_train, synth_train], [(1 - args.synth_weight), args.synth_weight])
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
        train, test = ds.split_synthetic_dataset(dataset, args.seed)
    return train, test
    

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
    elif dataset in ('srml2018', 'rml2018r', 'sm_rml2018'):
        return ds.SRML2018.classes
    elif dataset == 'rml2018':
        return RML2018_D.classes
    else:
        raise NotImplementedError(f"dataset not implemented: {dataset}")

def get_input_dims(args):
    pass


def create_dataloaders(args, trainset, testset):
    if args.n_batches != -1:
        if args.n_batches * args.batch_size > len(trainset):
            raise ValueError(f"n_batches * batch_size > len(trainset)")
        trainset = SubsetDataset(trainset, np.random.choice(len(trainset), args.n_batches * args.batch_size, replace=False))
    if args.dataset in ('sm_rml2018'):
        tr_sampler = BatchMixerSampler(trainset, args.batch_size)
    else:
        tr_sampler = RandomSampler(trainset)
    trainloader = DataLoader(
        trainset, batch_size=args.batch_size, sampler=tr_sampler, 
        num_workers=args.n_workers, pin_memory=True, persistent_workers=True)
    testloader = DataLoader(
        testset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.n_workers, pin_memory=True, persistent_workers=True)
    return trainloader, testloader