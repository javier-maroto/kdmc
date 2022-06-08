"""Custom RML2016.10A data functions"""
# pylint: disable=abstract-method,arguments-differ
import os
import scipy.io
from tqdm import tqdm

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset


def get_sawgn_datasets(path, time_samples=None, seed=0):

    full = SBasic(path, time_samples)
    full_test = SBasic(path, time_samples)

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
        'GFSK', 'CPFSK', 'B-FM', 'DSB-AM', 'SSB-AM', 'OQPSK'
    )
    time_samples = 1024
    filename = "sawgn.npz"

    def __init__(self, raw_path, time_samples=None):
        super().__init__()
        self.data_path = raw_path.joinpath(self.folder)
        if time_samples is not None:
            if time_samples > self.time_samples:
                print(
                    "Warning: cfg.time_samples too high for the dataset.",
                    f"Choosing {self.time_samples} instead.",
                )
            else:
                self.time_samples = time_samples
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}
        if not os.path.isfile(self.data_path.joinpath(self.filename)):
            self.create_dataset()
        self.iq, self.modulation, self.y, self.snr, self.snr_filt, self.sps, self.rolloff, self.fs = self.load()
        self.len = self.iq.shape[0]

    def compute_snr(self, signal, noise):
        """Computes the SNR of one/multiple signal (assumes last two dimensions are the 
        iq channels and time samples)"""
        if signal.shape != noise.shape:
            raise ValueError("Signal and noise must have the same shape")
        return 10 * np.log10(np.mean(signal ** 2, axis=(-1,-2)) / np.mean(noise ** 2, axis=(-1,-2)))

    def create_dataset(self):
        # Set seed for reproducibility
        np.random.seed(0)

        paths = []
        for _, _, files in os.walk(self.data_path.joinpath("rx_x")):
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
        
        df_filt = df_path.loc[
            (df_path.channel == 'AWGN') &
            (df_path.fs == 2e5) & (df_path.rolloff == 0.35) &
            df_path.modulation.isin(self.classes)]
        df_params = df_filt.groupby(['sps', 'rolloff']).count()
        n_params = df_params.shape[0]
        basic_paths = df_filt['path'].values
        print(df_filt[['fs','sps','rolloff']].drop_duplicates())
        modulation = []
        rx_x = []
        y = []
        snr = []
        snr_filt, sps_list, rolloff_list, fs_list = [], [], [], []
        for path in tqdm(basic_paths):
            # Load iq data
            data = scipy.io.loadmat(self.data_path.joinpath("rx_x", path))["rx_x"]
            n_signals = data.shape[0] // n_params  # Normalize to compare with other datasets
            idxs = np.random.choice(data.shape[0], n_signals, replace=False)
            data = data[idxs, :self.time_samples]
            assert data.shape[0] == n_signals, (data.shape, n_signals)
            data = np.stack([data.real, data.imag], axis=1)
            rx_x.append(data.astype(np.float32))
            # Load modulation
            mod = df_path.loc[df_path.path == path, 'modulation'].values[0]
            modulation.append(np.full(data.shape[0], self.class_to_idx[mod]))
            y.append(scipy.io.loadmat(self.data_path.joinpath("y", path))['y'][idxs])
            # Load sps
            sps = df_path.loc[df_path.path == path, 'sps'].values[0]
            sps_list.append(np.full(data.shape[0], sps))
            # Load rolloff
            rolloff = df_path.loc[df_path.path == path, 'rolloff'].values[0]
            rolloff_list.append(np.full(data.shape[0], rolloff))
            # Load fs
            fs = df_path.loc[df_path.path == path, 'fs'].values[0]
            fs_list.append(np.full(data.shape[0], fs))
            # Load snr
            snr_i = df_path.loc[df_path.path == path, 'snr'].values[0]
            snr.append(np.full(data.shape[0], snr_i))
            if mod in ['GFSK', 'CPFSK', 'B-FM', 'DSB-AM', 'SSB-AM','OQPSK']:
                snr_filt.append(np.full(data.shape[0], np.nan))
            else:
                # Load rx_s and tx_s
                data = scipy.io.loadmat(self.data_path.joinpath("rx_s", path))["rx_s"][idxs]
                data = np.stack([data.real, data.imag], axis=1)
                rx_s = data.astype(np.float32)
                data = scipy.io.loadmat(self.data_path.joinpath("tx_s", path))["tx_s"][idxs]
                data = np.stack([data.real, data.imag], axis=1)
                tx_s = data.astype(np.float32)
                snr_filt.append(self.compute_snr(tx_s, rx_s - tx_s))

        modulation = np.concatenate(modulation, axis=0, dtype=np.int64)
        snr = np.concatenate(snr, axis=0)
        rx_x = np.concatenate(rx_x, axis=0)
        y = np.concatenate(y, axis=0, dtype=np.float32)
        sps = np.concatenate(sps_list, axis=0)
        rolloff = np.concatenate(rolloff_list, axis=0)
        fs = np.concatenate(fs_list, axis=0)
        snr_filt = np.concatenate(snr_filt, axis=0)
        
        assert len(np.unique(modulation)) == 20, len(np.unique(modulation))
        assert y.shape[0] == snr_filt.shape[0], (y.shape, snr_filt.shape)
        np.savez(
            self.data_path.joinpath(self.filename),
            iq=rx_x, modulation=modulation, y=y, snr=snr, snr_filt=snr_filt,
            sps=sps, rolloff=rolloff, fs=fs)

    def load(self):
        """Returns the pytables arrays for the iq signals, modulations and snrs"""
        data = np.load(self.data_path.joinpath(self.filename))
        iq = data['iq']
        modulation = data['modulation']
        y = data['y']
        snr = data['snr']
        snr_filt = data['snr_filt']
        sps = data['sps']
        rolloff = data['rolloff']
        fs = data['fs']
        
        return iq, modulation, y, snr, snr_filt, sps, rolloff, fs

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return {
            "x": self.iq[idx],
            "y": self.y[idx],
            "snr": self.snr[idx],
            "snr_filt": self.snr_filt[idx],
            "idx": idx,
            "fs": self.fs[idx],
            "sps": self.sps[idx],
            "rolloff": self.rolloff[idx]
        }
