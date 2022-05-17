from foolbox import Attack
from numpy import dtype
import torch
from kdmc.attack.pgd import PGD
from kdmc.utils import FFloat


def parse_attack(net, atk_arg):
    if atk_arg[0] == 'pgd':
        norm = atk_arg[1]
        spr = FFloat(atk_arg[2])
        r_alpha = FFloat(atk_arg[3])
        steps = int(atk_arg[4])
        if norm == 'Linf':
            atk = PGD(net, eps=0, alpha=0, steps=steps)
            return SPR_Attack(atk, spr, r_alpha)
    

class SPR_Attack:

    def __init__(self, atk, spr, r_alpha) -> None:
        self.atk = atk
        self.spr = spr
        self.r_alpha = r_alpha

    def __call__(self, x, y, snr=None):
        noise_factor = 0 if snr is None else (1 / 10.0 ** (snr.to(dtype=torch.float32) / 10.0))
        signal_sample_energy = (x ** 2).sum(-2).mean(-1) / (1 + noise_factor)
        epsilon = ((signal_sample_energy / 2.0) / 10.0 ** (self.spr / 10.0)) ** 0.5
        self.atk.eps = epsilon.view(-1, 1, 1)
        self.atk.alpha = epsilon.view(-1, 1, 1) * self.r_alpha
        return self.atk(x, y)