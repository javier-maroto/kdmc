"""Custom RML2016.10A data functions"""
# pylint: disable=abstract-method,arguments-differ
import os
import pickle
import tarfile
from urllib.request import urlretrieve

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset
from tqdm import tqdm

from kdmc.data.utils import IndexableDataset
from kdmc.utils import _reporthook



def get_rml2016_10a_datasets(path, time_samples=None, return_idxs=False, seed=0):

    full = RML2016_10A_D(path, time_samples)

    keys = tuple(zip(full.modulation, full.snr))
    d = dict(zip(set(keys), range(len(full))))
    groups = np.array([d[x] for x in keys])
    train_idxs, test_idxs = train_test_split(
        np.arange(len(full)),
        test_size=0.3,
        random_state=seed,
        stratify=groups,
    )
    trainset = Subset(full, train_idxs)
    testset = Subset(full, test_idxs)

    return trainset, testset


class RML2016_10A_D(Dataset):
    """Dataset class"""

    remote_url = "http://opendata.deepsig.io/datasets/2016.10/RML2016.10a.tar.bz2"
    folder = "rml2016_10a"
    classes = (
        "WBFM",
        "AM-DSB",
        "AM-SSB",
        "CPFSK",
        "GFSK",
        "PAM4",
        "BPSK",
        "QPSK",
        "8PSK",
        "QAM16",
        "QAM64",
    )
    T_SAMPLES = 128

    def __init__(self, raw_path, time_samples=None):
        super().__init__()
        self.data_path = raw_path
        folder = os.path.join(self.data_path, self.folder)
        os.makedirs(folder, exist_ok=True)
        self.dataset_path = os.path.join(folder, "RML2016.10a_dict.pkl")
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

        if not os.path.isfile(self.dataset_path):
            self._download()

        self.iq, self.modulation, self.y, self.snr = self.load()
        self.len = self.iq.shape[0]

    def load(self):
        """Returns the pytables arrays for the iq signals, modulations and snrs"""
        with open(self.dataset_path, "rb") as infile:
            data = pickle.load(infile, encoding="latin")

        modulation = []
        snrs = []
        y = []
        iq = list()
        for key, value in data.items():
            num_iq_samples = value.shape[0]
            mod_idx = self.class_to_idx[key[0]]
            mod = np.full(num_iq_samples, mod_idx)
            y_item = np.zeros([num_iq_samples, len(self.classes)])
            y_item[:, mod_idx] = 1
            snr = np.full(num_iq_samples, key[1])
            modulation.append(mod)
            snrs.append(snr)
            iq.append(value[..., :self.time_samples])
            y.append(y_item)
            
        modulation = np.concatenate(modulation, axis=0, dtype=np.int64)
        snr = np.concatenate(snrs, axis=0)
        iq = np.concatenate(iq, axis=0)
        y = np.concatenate(y, axis=0)
        return iq, modulation, y, snr

    def _download(self):
        with tqdm(unit="B", unit_scale=True, miniters=1) as t:
            dpath = urlretrieve(self.remote_url, reporthook=_reporthook(t))[0]
        with tarfile.open(dpath, "r:bz2") as tar:
            tar.extractall(os.path.join(self.data_path, self.folder))
        os.remove(dpath)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return {
            "x": self.iq[idx],
            "y": self.y[idx],
            "snr": self.snr[idx],
            "idx": idx,
        }
