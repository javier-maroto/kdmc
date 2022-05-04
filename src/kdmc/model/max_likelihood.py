import numpy as np
import torch
import torch.nn.functional as F
from commpy.filters import rrcosfilter


class MaxLikelihoodModel:
    """
    This class is used to calculate the maximum likelihood function based on predefined modulations
    
    Currently, the following modulations are supported:
    BPSK, QPSK, 8-PSK, 16-APSK, 32-APSK, 64-APSK, 128-APSK, 256-APSK, 
    PAM4, 16-QAM, 32-QAM, 64-QAM, 128-QAM, 256-QAM
    
    """
    SUPPORTED_MODS = ["BPSK", "QPSK", "8-PSK",
        "16-APSK", "32-APSK", "64-APSK", "128-APSK", "256-APSK",
        "PAM4", "16-QAM", "32-QAM", "64-QAM", "128-QAM", "256-QAM",
    ]
    
    def __init__(self, modulations=None) -> None:
        if modulations is None:
            modulations = self.SUPPORTED_MODS
        else:
            for m in modulations:
                if m not in self.SUPPORTED_MODS:
                    raise ValueError(f"Modulation {m} not supported.")
        self.modulation_dict = dict(zip(range(len(modulations)), modulations))
        self.states_dict = self.load_states_dict()

    def load_states_dict(self):
        states_dict = {}
        for k, v in self.modulation_dict.items():
            states_dict[k] = self.load_states(v)
        return states_dict
    
    def load_states(self, modulation):
        pass

    def filter_received_signal(self, x, Ls, Ms, n_span_symb, rolloff):
        """Filters the received signal with a matched filter.
        Only valid for the supported modulations.

        Args:
            x (torch.Tensor): signal received. Shape: (batch_size, IQ, time_samples)
            Ls (int): number of samples per symbol.
            Ms (int): number of symbols per frame.
            n_span_symb (int): number of symbols to span the matched filter.
            rolloff (float): rolloff factor for the matched filter.
        
        Returns:
            torch.Tensor: filtered signal. Shape: (batch_size, IQ, time_samples)
        """
        # Create a filter with limited bandwidth. Parameters:
        #      N: Filter length in samples
        #    0.8: Roll off factor alpha
        #      1: Symbol period in time-units
        #     24: Sample rate in 1/time-units
        sPSF = rrcosfilter(n_span_symb * Ls, alpha=rolloff, Ts=1, Fs=over_sample)[1]

    def compute_ml(self, x, snr) -> torch.Tensor:
        """Constructs the likelihood function for the given SNR and samples per symbol.
        Only valid for the supported modulations.

        Args:
            x (torch.Tensor): signal received. Shape: (batch_size, IQ, time_samples)
            snr (float): SNR in dB.
        
        Returns:
            torch.Tensor: likelihood values ()
        """
        N0 = 10 ** (-snr/20)
        sigma = N0 / torch.sqrt(2)
        K = torch.log(2 * torch.pi * sigma ** 2)

        likelihood = torch.zeros(x.shape[0], len(self.states_dict))
        for j, mod_states in enumerate(self.states_dict.values()):
            M = len(mod_states)
            distances = torch.zeros((x.shape[0], x.shape[2], M))
            for i, state in enumerate(mod_states):
                distances[..., i] = torch.sum((x - state) ** 2, dim=1) / (2 * sigma ** 2)
            likelihood_sample = torch.logsumexp(-distances, dim=-1) + K
            likelihood[:, j] = torch.sum(likelihood_sample, dim=-1)
        return F.softmax(likelihood, dim=-1)
            
            

