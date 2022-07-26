"""Custom RML2016.10A data functions"""
# pylint: disable=abstract-method,arguments-differ
import os
import scipy.io
from tqdm import tqdm, trange

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset

from abc import abstractmethod, ABC

from kdmc.utils import compute_sample_energy


def split_synthetic_dataset(dataset, seed=0):
    """Split the synthetic dataset"""

    keys = tuple(zip(dataset.modulation, dataset.snr))
    d = dict(zip(set(keys), range(len(dataset))))
    groups = np.array([d[x] for x in keys])
    train_idxs, test_idxs = train_test_split(
        np.arange(len(dataset)),
        test_size=0.1,
        random_state=seed,
        stratify=groups,
    )
    trainset = Subset(dataset, train_idxs)
    testset = Subset(dataset, test_idxs)

    return trainset, testset


class Synthetic(Dataset, ABC):
    """Dataset class. They normally have 2.6M samples in total."""

    ALL_CLASSES = np.array([
        "BPSK", "QPSK", "8-PSK",
        "16-APSK", "32-APSK", "64-APSK", "128-APSK", "256-APSK",
        "PAM4", "16-QAM", "32-QAM", "64-QAM", "128-QAM", "256-QAM",
        'GFSK', 'CPFSK', 'B-FM', 'DSB-AM', 'SSB-AM', 'OQPSK'
    ])
    folder = "synthetic/signal"
    classes = ()
    time_samples = 1024
    filename = None

    def __init__(self, raw_path, time_samples=None, dataset_size=None):
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
        if dataset_size is not None:
            self.filename = f"{self.filename}_n={dataset_size}.npz"
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}
        self.reorder = np.array([np.nonzero(self.ALL_CLASSES == x) for x in self.classes])
        self.reorder = np.ravel(self.reorder)
        if not os.path.isfile(self.data_path.joinpath(self.filename)):
            print("Creating dataset...")
            self.create_dataset(dataset_size)
        else:
            print("Dataset exists in", self.data_path.joinpath(self.filename))
        self.iq, self.rx_s, self.modulation, self.y, self.snr, self.snr_filt, self.sps, self.rolloff, self.fs = self.load()
        self.len = self.iq.shape[0]
        

    def compute_snr(self, signal, noise):
        """Computes the SNR of one/multiple signal (assumes last two dimensions are the 
        iq channels and time samples)"""
        if signal.shape != noise.shape:
            raise ValueError("Signal and noise must have the same shape")
        return 10 * np.log10(np.mean(signal ** 2, axis=(-1,-2)) / np.mean(noise ** 2, axis=(-1,-2)))

    @abstractmethod
    def filter_paths(self, df_path):
        """Filter the possible communication scenarios based on the dataset specifications"""
        pass

    def create_dataset(self, dataset_size=None):
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
        
        df_res = self.filter_paths(df_path)
        if dataset_size is None:
            n_params = df_res.groupby(['channel', 'fs', 'sps', 'rolloff']).count().shape[0]
        else:
            n_params = (df_res.shape[0] * 10000) / dataset_size

        modulation = []
        rx_x, rx_s = [], []  # rx_s is the signal received as if there was only awgn noise
        y = []
        snr = []
        snr_filt, sps_list, rolloff_list, fs_list = [], [], [], []
        for i in trange(df_res.shape[0], desc="Creating dataset"):
            df1 = df_res.iloc[i]
            path = df1['path']
            # Load iq data
            data = scipy.io.loadmat(self.data_path.joinpath("rx_x", path))["rx_x"]
            n_signals = int(data.shape[0] / n_params)  # Normalize to compare with other datasets
            idxs = np.random.choice(data.shape[0], n_signals, replace=False)
            data = data[idxs, :self.time_samples]
            assert data.shape[0] == n_signals, (data.shape, n_signals)
            x = np.stack([data.real, data.imag], axis=1)
            rx_x.append(x.astype(np.float32))
            # Load modulation
            mod = df1['modulation']
            modulation.append(np.full(data.shape[0], self.class_to_idx[mod]))
            yp = scipy.io.loadmat(self.data_path.joinpath("y", path))['y'][idxs]
            yp = yp[:, self.reorder]
            y.append(yp)
            # Load sps
            sps = df1['sps']
            sps_list.append(np.full(data.shape[0], sps))
            # Load rolloff
            rolloff = df1['rolloff']
            rolloff_list.append(np.full(data.shape[0], rolloff))
            # Load fs
            fs = df1['fs']
            fs_list.append(np.full(data.shape[0], fs))
            # Load snr
            snr_i = df1['snr']
            snr.append(np.full(data.shape[0], snr_i))
            if mod in ['GFSK', 'CPFSK', 'B-FM', 'DSB-AM', 'SSB-AM','OQPSK']:
                snr_filt.append(np.full(x.shape[0], np.nan))
                rx_s.append(np.full_like(x, np.nan))
            else:
                # Load rx_s and tx_s
                data = scipy.io.loadmat(self.data_path.joinpath("rx_s", path))["rx_s"][idxs]
                data = np.stack([data.real, data.imag], axis=1)
                rs = data.astype(np.float32)
                data = scipy.io.loadmat(self.data_path.joinpath("tx_s", path))["tx_s"][idxs]
                data = np.stack([data.real, data.imag], axis=1)
                ts = data.astype(np.float32)
                snr_filt.append(self.compute_snr(ts, rs - ts))
                rs_extended = np.zeros_like(x)
                n_symb = int(x.shape[-1] / sps)
                rs_extended[..., :n_symb] = rs[..., :n_symb]
                rx_s.append(rs_extended)

        modulation = np.concatenate(modulation, axis=0, dtype=np.int64)
        snr = np.concatenate(snr, axis=0)
        rx_x = np.concatenate(rx_x, axis=0)
        rx_s = np.concatenate(rx_s, axis=0)
        y = np.concatenate(y, axis=0, dtype=np.float32)
        sps = np.concatenate(sps_list, axis=0)
        rolloff = np.concatenate(rolloff_list, axis=0)
        fs = np.concatenate(fs_list, axis=0)
        snr_filt = np.concatenate(snr_filt, axis=0)
        
        assert y.shape[0] == snr_filt.shape[0], (y.shape, snr_filt.shape)
        np.savez(
            self.data_path.joinpath(self.filename),
            iq=rx_x, rx_s=rx_s, modulation=modulation, y=y, snr=snr, snr_filt=snr_filt,
            sps=sps, rolloff=rolloff, fs=fs)

    def load(self, dataset_size=None):
        """Returns the pytables arrays for the iq signals, modulations and snrs"""
        data = np.load(os.path.join(self.data_path, self.filename))
        iq = data['iq'][..., :self.time_samples]
        rx_s = data['rx_s'][..., :self.time_samples]
        modulation = data['modulation']
        y = data['y']
        snr = data['snr']
        snr_filt = data['snr_filt']
        sps = data['sps']
        fs = data['fs']
        rolloff = data['rolloff']
        return iq, rx_s, modulation, y, snr, snr_filt, sps, rolloff, fs

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return {
            "x": self.iq[idx],
            "x_ml": self.rx_s[idx],
            "y": self.y[idx],
            "snr": self.snr[idx],
            "snr_filt": self.snr_filt[idx],
            "idx": idx,
            "fs": self.fs[idx],
            "sps": self.sps[idx],
            "rolloff": self.rolloff[idx]
        }


class SAWGNp0c20(Synthetic):
    """Synthetic dataset with basic parameters"""

    classes = (
        "BPSK", "QPSK", "8-PSK",
        "16-APSK", "32-APSK", "64-APSK", "128-APSK", "256-APSK",
        "PAM4", "16-QAM", "32-QAM", "64-QAM", "128-QAM", "256-QAM",
        'GFSK', 'CPFSK', 'B-FM', 'DSB-AM', 'SSB-AM', 'OQPSK'
    )
    filename = "sawgn_p0c20.npz"

    def filter_paths(self, df_path):
        df_res = df_path.loc[
            (df_path.sps == 8) & (df_path.channel == 'AWGN') &
            (df_path.fs == 2e5) & (df_path.rolloff == 0.35) &
            df_path.modulation.isin(self.classes)]
        return df_res

class SAWGNp1c20(Synthetic):
    """Synthetic dataset with basic parameters"""

    classes = (
        "BPSK", "QPSK", "8-PSK",
        "16-APSK", "32-APSK", "64-APSK", "128-APSK", "256-APSK",
        "PAM4", "16-QAM", "32-QAM", "64-QAM", "128-QAM", "256-QAM",
        'GFSK', 'CPFSK', 'B-FM', 'DSB-AM', 'SSB-AM', 'OQPSK'
    )
    filename = "sawgn_p1c20.npz"

    def filter_paths(self, df_path):
        df_res = df_path.loc[
            (df_path.channel == 'AWGN') &
            (df_path.fs == 2e5) & (df_path.rolloff == 0.35) &
            df_path.modulation.isin(self.classes)]
        return df_res


class SAWGNp2c20(Synthetic):
    """Synthetic dataset with basic parameters"""

    classes = (
        "BPSK", "QPSK", "8-PSK",
        "16-APSK", "32-APSK", "64-APSK", "128-APSK", "256-APSK",
        "PAM4", "16-QAM", "32-QAM", "64-QAM", "128-QAM", "256-QAM",
        'GFSK', 'CPFSK', 'B-FM', 'DSB-AM', 'SSB-AM', 'OQPSK'
    )
    filename = "sawgn_p2c20.npz"

    def filter_paths(self, df_path):
        df_res = df_path.loc[
            (df_path.channel == 'AWGN') &
            (df_path.fs == 2e5) &
            df_path.modulation.isin(self.classes)]
        return df_res


class Sp0c20(Synthetic):
    """Synthetic dataset with basic parameters"""

    classes = (
        "BPSK", "QPSK", "8-PSK",
        "16-APSK", "32-APSK", "64-APSK", "128-APSK", "256-APSK",
        "PAM4", "16-QAM", "32-QAM", "64-QAM", "128-QAM", "256-QAM",
        'GFSK', 'CPFSK', 'B-FM', 'DSB-AM', 'SSB-AM', 'OQPSK'
    )
    filename = "s_p0c20.npz"

    def filter_paths(self, df_path):
        df_res = df_path.loc[
            (df_path.sps == 8) &
            (df_path.fs == 2e5) & (df_path.rolloff == 0.35) &
            df_path.modulation.isin(self.classes)]
        return df_res


class SRML2016_10A(Synthetic):
    """Synthetic dataset with basic parameters"""

    classes = (
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
    filename = "sawgn_p2c11.npz"
    time_samples = 128

    def __init__(self, raw_path, dataset_size=None):
        super().__init__(raw_path, None, dataset_size)
    
    def filter_paths(self, df_path):
        df_res = df_path.loc[
            (df_path.fs == 2e5) &
            df_path.modulation.isin(self.classes)]
        return df_res


class SRML2018(Synthetic):
    """Synthetic dataset with basic parameters"""

    classes = (
        "BPSK",
        "QPSK",
        "8-PSK",
        "16-PSK",
        "32-PSK",
        "16-APSK",
        "32-APSK",
        "64-APSK",
        "128-APSK",
        "16-QAM",
        "32-QAM",
        "64-QAM",
        "128-QAM",
        "256-QAM"
    )
    filename = "srml2018.mat"
    time_samples = 1024

    def __init__(self, raw_path, dataset_size=None):
        super(Dataset).__init__()
        self.data_path = raw_path.joinpath(self.folder)
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}
        self.iq, self.rx_s, self.modulation, self.y, self.snr, self.snr_filt, self.sps, self.rolloff, self.fs, self.channel = self.load()
        self.len = self.iq.shape[0]
    
    def load(self, dataset_size=None):
        import h5py
        with h5py.File(self.data_path.joinpath(self.filename), 'r') as f:
            rx_x = f['rx_x'][:]
            rx_x = np.swapaxes(rx_x, 0, 2)
            rx_x = rx_x.astype(np.float32)
            # Normalize rx_x
            rx_x = rx_x / compute_sample_energy(rx_x)[:, np.newaxis, np.newaxis]
            tx_s = f['tx_s'][:]
            tx_s = np.swapaxes(tx_s, 0, 2)
            rx_s = f['rx_s'][:]
            rx_s = np.swapaxes(rx_s, 0, 2)
            y = f['y'][:]
            y = np.swapaxes(y, 0, 1)
            y = y.astype(np.float32)

            modulation = np.argmax(y, axis=1)
            modulation = modulation.astype(np.int64)

            snr = np.squeeze(f['snrs'][:])
            sps = np.squeeze(f['lsps'][:])
            rolloff = np.squeeze(f['rolloffs'][:])
            fs = np.squeeze(f['fss'][:])
            channel = np.squeeze(f['rays'][:])
        snr_filt = self.compute_snr(tx_s, rx_s - tx_s)
        return rx_x, rx_s, modulation, y, snr, snr_filt, sps, rolloff, fs, channel

    def filter_paths(self, df_path):
        pass