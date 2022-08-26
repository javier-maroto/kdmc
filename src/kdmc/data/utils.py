from torch.utils.data import Dataset, ConcatDataset, Subset
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
        datasets = [Subset(d, np.random.choice(len(d), size=int(len(d)*w))) for d, w in zip(datasets, mix_weights)]
        self.dataset = ConcatDataset(datasets)

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)