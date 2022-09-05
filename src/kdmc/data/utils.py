from torch.utils.data import Dataset, ConcatDataset, Subset, Sampler
import torch
import numpy as np


class BasicDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.y)


class IndexableDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __getitem__(self, index):
        data, target = self.dataset[index]
        return data, target, index

    def __len__(self):
        return len(self.dataset)


class SubsetDataset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
        self.time_samples = dataset.time_samples

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self.dataset[[self.indices[i] for i in idx]]
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


class DatasetMixer(Dataset):
    def __init__(self, datasets, mix_weights):
        assert len(datasets) == len(mix_weights)
        self.time_samples = datasets[0].time_samples
        self.datasets = [Subset(d, np.random.choice(len(d), size=int(len(d)*w))) for d, w in zip(datasets, mix_weights)]
        self.dataset = ConcatDataset(datasets)

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


class BatchMixerSampler(Sampler[int]):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Args:
        indices (sequence): a sequence of indices
        generator (Generator): Generator used in sampling.
    """
    def __init__(self, mix_dataset: ConcatDataset, batch_size, generator=None) -> None:
        self.mix_dataset = mix_dataset
        self.generator = generator
        self.batch_size = batch_size
        self.len = 0
        for dataset in self.mix_dataset.datasets:
            self.len += (len(dataset) // self.batch_size) * self.batch_size

    def __iter__(self):
        n2 = 0
        indices = []
        dataset_indices = []
        dataset_batches = []
        for didx, dataset in enumerate(self.mix_dataset.datasets):
            n_batches = len(dataset) // self.batch_size
            n = n_batches * self.batch_size
            indices.append(torch.randperm(len(dataset))[:n] + n2)
            dataset_indices.append(torch.full([n_batches], didx))
            dataset_batches.append(torch.arange(n_batches))
            n2 += len(dataset)
        dataset_indices = torch.cat(dataset_indices)
        dataset_batches = torch.cat(dataset_batches)
        perm = torch.randperm(len(dataset_batches))
        dataset_indices = dataset_indices[perm]
        dataset_batches = dataset_batches[perm]
        for i in range(len(dataset_batches)):
            for j in range(self.batch_size):
                dindices = indices[dataset_indices[i]]
                yield dindices[dataset_batches[i] * self.batch_size + j]

    def __len__(self) -> int:
        return self.len