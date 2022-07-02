import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io
import os

FILTERS = {
    "sps-8_span-8_rc-0.35": [0.000255413507322333,0.00110422182418781,0.00166527642963825,0.00172728942499136,0.00119672217082238,0.000143118377136952,-0.00119380933122976,-0.00244282808313376,-0.00318217479899278,-0.00304811162353564,-0.00184716549701300,0.000356156769634307,0.00320234275056614,0.00606806822478606,0.00816924442299973,0.00872309340615226,0.00714095795073857,0.00321272897261628,-0.00275948306981454,-0.00991503060643535,-0.0168979866895801,-0.0220315921582807,-0.0235825830725101,-0.0200735499350221,-0.0105878322047461,0.00499197737985832,0.0258626845030293,0.0503282966092729,0.0759828182924085,0.100025226547374,0.119658253120396,0.132505285668751,0.136974268964650,0.132505285668751,0.119658253120396,0.100025226547374,0.0759828182924085,0.0503282966092729,0.0258626845030293,0.00499197737985832,-0.0105878322047461,-0.0200735499350221,-0.0235825830725101,-0.0220315921582807,-0.0168979866895801,-0.00991503060643535,-0.00275948306981454,0.00321272897261628,0.00714095795073857,0.00872309340615226,0.00816924442299973,0.00606806822478606,0.00320234275056614,0.000356156769634307,-0.00184716549701300,-0.00304811162353564,-0.00318217479899278,-0.00244282808313376,-0.00119380933122976,0.000143118377136952,0.00119672217082238,0.00172728942499136,0.00166527642963825,0.00110422182418781,0.000255413507322333],
}


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
    
    def __init__(self, states_path, modulations, device=None) -> None:
        self.device = device
        self.supported = torch.full([len(modulations)], False, dtype=torch.bool)
        supp_mods = []
        for mi, m in enumerate(modulations):
            if m in self.SUPPORTED_MODS:
                self.supported[mi] = True
                supp_mods.append(m)
        self.modulation_dict = dict(zip(range(len(supp_mods)), supp_mods))
        self.states_dict = self.load_states_dict(states_path)
        self.h = None
        self.Ls = None

    def load_states_dict(self, states_path=None) -> dict:
        states_dict = {}
        for k, v in self.modulation_dict.items():
            states_dict[k] = self.load_states(states_path, v)
        return states_dict
    
    def load_states(self, states_path, modulation):
        states = scipy.io.loadmat(os.path.join(states_path, f"{modulation}.mat"))["states"]
        return torch.tensor(states, device=self.device)

    def create_filter(self, Ls=8, Ms=1, n_span_symb=8, rolloff=0.35):
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
        self.Ls = Ls
        self.n_span_symb = n_span_symb
        # Used extracted MATLAB coeffs
        if (rolloff == 0.35) and (Ls == 8) and (Ms == 1) and (n_span_symb == 8):
            self.h = torch.tensor(FILTERS["sps-8_span-8_rc-0.35"], dtype=torch.float32, device=self.device)
        # Almost exact copy of the above. TODO: Add Ms to the filter construction
        else:
            N = Ls * n_span_symb + 1
            t_vector = np.arange(N)-(N-1)/2
            h = np.zeros(N, dtype=np.float64)

            for x, t in enumerate(t_vector):
                if t == 0.0:
                    h[x] = 1.0 - rolloff + (4*rolloff/np.pi)
                elif rolloff != 0 and abs(t) == Ls/(4*rolloff):
                    h[x] = (rolloff/np.sqrt(2))*(((1+2/np.pi)* \
                            (np.sin(np.pi/(4*rolloff)))) + ((1-2/np.pi)*(np.cos(np.pi/(4*rolloff)))))
                else:
                    h[x] = (np.sin(np.pi*t*(1-rolloff)/Ls) +  \
                            4*rolloff*(t/Ls)*np.cos(np.pi*t*(1+rolloff)/Ls))/ \
                            (np.pi*t*(1-(4*rolloff*t/Ls)*(4*rolloff*t/Ls))/Ls)
            
            h /= np.sum(h)
            self.h = torch.from_numpy(h).to(self.device, dtype=torch.float32)
        self.h = self.h.view(1,1,-1).repeat(2,1,1)

    def filter_signal(self, x):
        """Filters the received signal with a matched filter.
        Only valid for the supported modulations.

        Args:
            x (torch.Tensor): signal received. Shape: (batch_size, IQ, time_samples)
        
        Returns:
            torch.Tensor: filtered signal. Shape: (batch_size, IQ, time_samples)
        """
        x = torch.conv1d(x, self.h, groups=2)
        # Downsample
        x = x[..., ::self.Ls]
        return x

    def compute_ml(self, x, snr, use_filter=True) -> torch.Tensor:
        """Constructs the likelihood function for the given SNR and samples per symbol.
        Only valid for the supported modulations.

        Args:
            x (torch.Tensor): signal received. Shape: (batch_size, IQ, time_samples)
            snr (float): SNR in dB.
        
        Returns:
            torch.Tensor: likelihood values ()
        """
        if use_filter:
            if self.h is None:
                raise ValueError("Filter not created.")
            x = self.filter_signal(x)
        N0 = 10 ** (-snr/20)
        sigma = N0 / np.sqrt(2)
        Ks = -torch.log(2 * torch.pi * sigma ** 2).view(-1, 1)
        likelihood = torch.zeros(x.shape[0], len(self.states_dict), device=self.device)
        for j, mod_states in self.states_dict.items():
            M = len(mod_states)
            Km = -np.log(M)
            distances = torch.zeros((x.shape[0], x.shape[2], M), device=self.device)
            for i, state in enumerate(mod_states):
                s = state.view([1, -1, 1])
                distances[..., i] = torch.sum((x - s) ** 2, dim=1)
            likelihood_sample = torch.logsumexp(-distances / (2 * sigma.view(-1, 1, 1) ** 2), dim=-1) + Ks + Km
            likelihood[:, j] = torch.sum(likelihood_sample, dim=-1)
        ml_preds = F.softmax(likelihood, dim=-1)
        return ml_preds

    def compute_ml_symb(self, x, snr, sps=1) -> torch.Tensor:
        """Constructs the likelihood function for the given SNR and samples per symbol.
        Only valid for the supported modulations.

        Args:
            x (torch.Tensor): signal received. Shape: (batch_size, IQ, time_samples)
            snr (float): SNR in dB.
        
        Returns:
            torch.Tensor: likelihood values ()
        """
        x = x[..., :(x.shape[-1] // sps)]
        N0 = 10 ** (-snr/20)
        sigma = N0 / np.sqrt(2)
        Ks = -torch.log(2 * torch.pi * sigma ** 2).view(-1, 1)
        likelihood = torch.zeros(x.shape[0], len(self.states_dict), device=self.device)
        for j, mod_states in self.states_dict.items():
            M = len(mod_states)
            Km = -np.log(M)
            distances = torch.zeros((x.shape[0], x.shape[2], M), device=self.device)
            for i, state in enumerate(mod_states):
                s = state.view([1, -1, 1])
                distances[..., i] = torch.sum((x - s) ** 2, dim=1)
            likelihood_sample = torch.logsumexp(-distances / (2 * sigma.view(-1, 1, 1) ** 2), dim=-1) + Ks + Km
            likelihood[:, j] = torch.sum(likelihood_sample, dim=-1)
        ml_preds = F.softmax(likelihood, dim=-1)
        return ml_preds

    def compute_advml(self, x, x_adv, snr, use_filter=True) -> torch.Tensor:
        """Constructs the likelihood function for the given SNR and samples per symbol.
        Takes into account the adversarial noise.
        Only valid for the supported modulations.

        Args:
            x (torch.Tensor): signal received. Shape: (batch_size, IQ, time_samples)
            snr (float): SNR in dB.
        
        Returns:
            torch.Tensor: likelihood values ()
        """
        if use_filter:
            if self.h is None:
                raise ValueError("Filter not created.")
            x = self.filter_signal(x)
            x_adv = self.filter_signal(x_adv)
            sigma_adv = torch.sqrt(torch.mean((x_adv - x) ** 2))
        N0 = 10 ** (-snr/20)
        sigma = N0 / np.sqrt(2)
        sigma = torch.sqrt(sigma ** 2 + sigma_adv ** 2)
        Ks = -torch.log(2 * torch.pi * sigma ** 2).view(-1, 1)
        likelihood = torch.zeros(x.shape[0], len(self.states_dict), device=self.device)
        for j, mod_states in self.states_dict.items():
            M = len(mod_states)
            Km = -np.log(M)
            distances = torch.zeros((x.shape[0], x.shape[2], M), device=self.device)
            for i, state in enumerate(mod_states):
                s = state.view([1, -1, 1])
                distances[..., i] = torch.sum((x - s) ** 2, dim=1)
            likelihood_sample = torch.logsumexp(-distances / (2 * sigma.view(-1, 1, 1) ** 2), dim=-1) + Ks + Km
            likelihood[:, j] = torch.sum(likelihood_sample, dim=-1)
        ml_preds = F.softmax(likelihood, dim=-1)
        return ml_preds

    def adapt_unsupported(self, ml_preds, targets):
        """Adapts the likelihood function for the unsupported modulations.
        Assumes all non-supported modulations are at the last indexes.

        Args:
            ml_preds (torch.Tensor): likelihood values. Shape: (batch_size, ml_targets)
            targets (torch.Tensor): target values. Shape: (batch_size, targets)
        
        Returns:
            torch.Tensor: likelihood values ()
        """
        res = torch.zeros(ml_preds.shape[0], len(self.supported), device=self.device)
        mask = (targets * self.supported.to(self.device)).sum(-1) == 1
        res[:, self.supported] = ml_preds
        res[mask] = targets[mask]
        return res

    def return_ml_model(self, snr=None):
        """Returns the model.
        """
        class MLModel(nn.Module):
            def __init__(self, ml_model, snr=None):
                super().__init__()
                self.h = nn.Parameter(ml_model.h.detach())
                self.states_dict = ml_model.states_dict
                self.device = ml_model.device
                self.snr = snr
                self.Ls = ml_model.Ls

            def forward(self, x):
                x = torch.conv1d(x, self.h, groups=2)
                # Downsample
                x = x[..., ::self.Ls]
                N0 = 10 ** (-self.snr/20)
                sigma = N0 / np.sqrt(2)
                Ks = -torch.log(2 * torch.pi * sigma ** 2).view(-1, 1)
                likelihood = torch.zeros(x.shape[0], len(self.states_dict), device=self.device)
                for j, mod_states in self.states_dict.items():
                    M = len(mod_states)
                    Km = -np.log(M)
                    distances = torch.zeros((x.shape[0], x.shape[2], M), device=self.device)
                    for i, state in enumerate(mod_states):
                        s = state.view([1, -1, 1])
                        distances[..., i] = torch.sum((x - s) ** 2, dim=1)
                    likelihood_sample = torch.logsumexp(-distances / (2 * sigma.view(-1, 1, 1) ** 2), dim=-1) + Ks + Km
                    likelihood[:, j] = torch.sum(likelihood_sample, dim=-1)
                ml_preds = F.softmax(likelihood, dim=-1)
                return ml_preds
        
        return MLModel(self, snr)
            
            

