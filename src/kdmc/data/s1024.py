"""Custom RML2016.10A data functions"""
# pylint: disable=abstract-method,arguments-differ
import os
import h5py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset
from tqdm import tqdm


def get_s1024_datasets(path, time_samples=None, seed=0):

    full = S1024(path, time_samples)

    keys = tuple(zip(full.modulation, full.snr))
    d = dict(zip(set(keys), range(len(full))))
    groups = np.array([d[x] for x in keys])
    train_idxs, test_idxs = train_test_split(
        np.arange(len(full)),
        test_size=0.1,
        random_state=seed,
        stratify=groups,
    )
    trainset = Subset(full, train_idxs)
    testset = Subset(full, test_idxs)

    return trainset, testset


class S1024(Dataset):
    """Dataset class"""

    folder = "synthetic/1024"
    classes = (
        "BPSK","QPSK","OQPSK","PSK8","QAM16","QAM32",
        "APSK16","APSK32","APSK64","APSK128", "APSK256",
        "QAM64","QAM128","QAM256","PAM4","GFSK","CPFSK","BFM","DSBAM","SSBAM"
    )
    T_SAMPLES = 1024

    def __init__(self, raw_path, time_samples=None):
        super().__init__()
        self.data_path = os.path.join(raw_path, self.folder)
        self.time_samples = self.T_SAMPLES
        if time_samples is not None:
            if time_samples > self.T_SAMPLES:
                print(
                    "Warning: cfg.time_samples too high for the dataset.",
                    f"Choosing {self.T_SAMPLES} instead.",
                )
            else:
                self.time_samples = time_samples
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

        self.iq, self.modulation, self.snr = self.load()
        self.len = self.iq.shape[0]

    def load(self):
        """Returns the pytables arrays for the iq signals, modulations and snrs"""
        df_keys = pd.read_csv(os.path.join(self.data_path, "dataset1024_keys.csv"))
        sps8_keys = df_keys.loc[(df_keys.L == 8) & (df_keys.M == 1), 'key'].values

        modulation = []
        snr = []
        iq = list()
        with h5py.File(os.path.join(self.data_path,'dataset1024.mat')) as f:
            for key in f['ds'].keys():
                if key in sps8_keys:
                    data = f['ds'][key][:self.time_samples, :2]
                    iq.append(data.astype(np.float32))
                    mod = df_keys.loc[df_keys.key == key, 'mod'].values[0]
                    modulation.append(np.full(data.shape[2], self.class_to_idx[mod]))
                    snr_i = df_keys.loc[df_keys.key == key, 'snr'].values[0]
                    snr.append(np.full(data.shape[2], snr_i))

        modulation = np.concatenate(modulation, axis=0, dtype=np.int64)
        snr = np.concatenate(snr, axis=0)
        iq = np.concatenate(iq, axis=2)
        iq = iq.reshape(iq.shape[2], iq.shape[1], iq.shape[0])
        return iq, modulation, snr

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return {
            "x": self.iq[idx],
            "y": self.modulation[idx],
            "snr": self.snr[idx],
            "idx": idx,
        }
