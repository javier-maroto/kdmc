"""Custom RML2018 data functions"""
# pylint: disable=abstract-method,arguments-differ
import os
import tarfile
from urllib.request import urlretrieve

import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

import tables
from kdmc.data.utils import SubsetDataset

from kdmc.utils import _reporthook


def get_rml2018_datasets(path, time_samples=None, return_idxs=False, seed=0):

    full = RML2018_D(path, time_samples)

    keys = tuple(zip(full.modulation, full.snr))
    d = dict(zip(set(keys), range(len(full))))
    groups = np.array([d[x] for x in keys])
    train_idxs, test_idxs = train_test_split(
        np.arange(len(full)),
        test_size=0.3,
        random_state=seed,
        stratify=groups,
    )
    trainset = SubsetDataset(full, train_idxs)
    testset = SubsetDataset(full, test_idxs)

    return trainset, testset


class RML2018_D(Dataset):
    """Dataset class. There are 2555904 records of 1024 samples long (B x 1024 x 2).

    The 'classes.txt' file ordering is incorrect. Below is the most probable correct one.
    Source:
https://cyclostationary.blog/2020/09/24/deepsigs-2018-data-set-2018-01-osc-0001_1024x2m-h5-tar-gz/
    """
    remote_url = "http://opendata.deepsig.io/datasets/2018.01/2018.01.OSC.0001_1024x2M.h5.tar.gz"
    classes = (
        'OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK', '16APSK', '32APSK',
        '64APSK', '128APSK', '16QAM', '32QAM', '64QAM', '128QAM', '256QAM', 'AM-SSB-WC',
        'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC', 'FM', 'GMSK', 'OQPSK')
    folder = "2018.01"
    T_SAMPLES = 1024

    def __init__(self, raw_path, time_samples=None):
        super().__init__()
        self.data_path = raw_path
        folder = os.path.join(self.data_path, self.folder)
        os.makedirs(folder, exist_ok=True)
        self.dataset_path = os.path.join(folder, 'GOLD_XYZ_OSC.0001_1024.hdf5')
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
        with tables.open_file(self.dataset_path, "r") as f:
            modulation = f.get_node("/", "Y")[:]
            snr = f.get_node("/", "Z")[:]
            iq = f.get_node("/", "X")[:]
        iq = np.swapaxes(iq, -1, -2)[..., :self.time_samples]
        y = np.argmax(modulation, axis=-1)
        snr = np.squeeze(snr)
        return iq, modulation, y, snr

    def _download(self):
        with tqdm(unit='B', unit_scale=True, miniters=1) as t:
            dpath = urlretrieve(self.remote_url, reporthook=_reporthook(t))[0]
        with tarfile.open(dpath, "r:gz") as tar:
            tar.extractall(self.data_path)
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
