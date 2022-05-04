"""Custom RML2016.10A data functions"""
# pylint: disable=abstract-method,arguments-differ
import os
import scipy.io
from tqdm import tqdm

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset


def get_sbasic_datasets(path, time_samples=None, seed=0, return_ml=False, use_filters=True):

    full = SBasic(path, time_samples, return_ml, use_filters)
    full_test = SBasic(path, time_samples, return_ml, use_filters)

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
    testset = Subset(full_test, test_idxs)

    return trainset, testset


class SBasic(Dataset):
    """Dataset class"""

    folder = "synthetic/signal"
    classes = (
        "BPSK", "QPSK", "8-PSK",
        "16-APSK", "32-APSK", "64-APSK", "128-APSK", "256-APSK",
        "PAM4", "16-QAM", "32-QAM", "64-QAM", "128-QAM", "256-QAM",
        "GFSK", "CPFSK", "OQPSK", "B-FM", "DSB-AM", "SSB-AM"
    )

    def __init__(self, raw_path, time_samples=None, return_ml=False, use_filters=True):
        super().__init__()
        self.data_path = os.path.join(raw_path, self.folder)
        self.time_samples = 1024 if use_filters else 128
        if time_samples is not None:
            if time_samples > self.time_samples:
                print(
                    "Warning: cfg.time_samples too high for the dataset.",
                    f"Choosing {self.time_samples} instead.",
                )
            else:
                self.time_samples = time_samples
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}
        file = "sbasic.npz" if use_filters else "sbasic_nf.npz"
        if not os.path.isfile(os.path.join(self.data_path, file)):
            self.create_dataset(use_filters)
        self.iq, self.modulation, self.y, self.snr = self.load(return_ml, use_filters)
        self.len = self.iq.shape[0]

    def create_dataset(self, use_filters=True):
        paths = []
        for _, _, files in os.walk(os.path.join(self.data_path, "rx_x")):
            for file in files:
                if file.endswith(".mat"):
                    paths.append(file)
        df_path = pd.DataFrame({'path': paths})
        df_path = pd.concat([
            df_path, df_path['path'].str.extract(
                r'(?P<channel>[\w]+)_(?P<modulation>[\w-]+)_(?P<fs>[\w.+-]+)_(?P<sps>[\w.+-]+)_(?P<rolloff>[\w.+-]+)_(?P<snr>[\w.+-]+)\.mat')], axis=1)
        df_path['fs'] = df_path['fs'].astype(float)
        df_path['sps'] = df_path['sps'].astype(float)
        df_path['rolloff'] = df_path['rolloff'].astype(float)
        df_path['snr'] = df_path['snr'].astype(float)
        
        basic_paths = df_path.loc[
            (df_path.sps == 8) & (df_path.channel == 'AWGN') &
            (df_path.fs == 2e5) & (df_path.rolloff == 0.35), 'path'].values

        modulation = []
        y = []
        snr = []
        yml = []
        yml_nf = []
        iq = list()
        for path in tqdm(basic_paths):
            data_key = "rx_x" if use_filters else "rx_nf"
            data = scipy.io.loadmat(os.path.join(self.data_path, data_key, path))[data_key]
            if data.shape[0] < self.time_samples:
                data = scipy.io.loadmat(os.path.join(self.data_path, "rx_x", path))["rx_x"]
            data = data[...,:self.time_samples]
            data = np.stack([data.real, data.imag], axis=1)
            iq.append(data.astype(np.float32))
            mod = df_path.loc[df_path.path == path, 'modulation'].values[0]
            modulation.append(np.full(data.shape[0], self.class_to_idx[mod]))
            yml.append(scipy.io.loadmat(os.path.join(self.data_path, "yml_est", path))['yml_est'])
            yml_nf.append(scipy.io.loadmat(os.path.join(self.data_path, "yml_nf", path))['yml_nf'])
            y.append(scipy.io.loadmat(os.path.join(self.data_path, "y", path))['y'])
            snr_i = df_path.loc[df_path.path == path, 'snr'].values[0]
            snr.append(np.full(data.shape[0], snr_i))

        modulation = np.concatenate(modulation, axis=0, dtype=np.int64)
        snr = np.concatenate(snr, axis=0)
        iq = np.concatenate(iq, axis=0)
        y = np.concatenate(y, axis=0, dtype=np.float32)
        yml = np.concatenate(yml, axis=0, dtype=np.float32)
        yml_nf = np.concatenate(yml_nf, axis=0, dtype=np.float32)
        
        file = "sbasic.npz" if use_filters else "sbasic_nf.npz"
        np.savez(
            os.path.join(self.data_path, file),
            iq=iq, modulation=modulation, y=y, snr=snr, yml=yml, yml_nf=yml_nf)

    def load(self, return_ml=False, use_filters=True):
        """Returns the pytables arrays for the iq signals, modulations and snrs"""
        file = "sbasic.npz" if use_filters else "sbasic_nf.npz"
        data = np.load(os.path.join(self.data_path, file))
        iq = data['iq']
        modulation = data['modulation']
        if return_ml:
            if use_filters:
                y = data['yml']
            else:
                y = data['yml_nf']
        else:
            y = data['y']
        snr = data['snr']
        
        return iq, modulation, y, snr

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return {
            "x": self.iq[idx],
            "y": self.y[idx],
            "snr": self.snr[idx],
            "idx": idx,
        }
