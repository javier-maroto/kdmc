from functools import partial
import torch
from tqdm import tqdm
import wandb

from kdmc.attack.core import SPR_Attack
from kdmc.attack.deepfool import DeepFool
from kdmc.attack.pgd import PGD


def get_acc_metrics(df, atk_key=None):
    """
    Compute the accuracy metrics for the given attack.
    """
    # Compute the accuracy metrics
    if atk_key is None:
        col = 'clean'
        label = ''
    else:
        col = atk_key
        label = atk_key + '_'
    temp = df.copy()
    temp['acc'] = temp[col] == temp['true']
    temp['acc_ml'] = temp[f'{col}_ml'] == temp[col]
    df_snr = temp.groupby('snr', as_index=False)[['acc','acc_ml']].mean().sort_values('snr')
    table = wandb.Table(data=df_snr, columns = ["snr", "acc", "acc_ml"])
    wandb.log({
        f"test.{label}acc": 100 * temp['acc'].mean(),
        f"test.{label}acc_ml": 100 * temp['acc_ml'].mean(),
        f"test.{label}acc_snr": wandb.plot.line(table, "snr", "acc", title=f"Accuracy vs SNR ({label})"),
        f"test.{label}acc_ml_snr": wandb.plot.line(table, "snr", "acc_ml", title=f"Accuracy vs SNR ({label})"),
        })

def get_acc_metrics_wo_ml(df, atk_key=None):
    """
    Compute the accuracy metrics for the given attack.
    """
    # Compute the accuracy metrics
    if atk_key is None:
        col = 'clean'
        label = ''
    else:
        col = atk_key
        label = atk_key + '_'
    temp = df.copy()
    temp['acc'] = temp[col] == temp['true']
    df_snr = temp.groupby('snr', as_index=False)[['acc']].mean().sort_values('snr')
    table = wandb.Table(data=df_snr, columns = ["snr", "acc"])
    wandb.log({
        f"test.{label}acc": 100 * temp['acc'].mean(),
        f"test.{label}acc_snr": wandb.plot.line(table, "snr", "acc", title=f"Accuracy vs SNR ({label})"),
        })

def measure_cosine_similarity(net1, net2, x, y, snr):
    atk1 = SPR_Attack(PGD(net1, steps=7), 20, 0.25)
    atk2 = SPR_Attack(PGD(net2, steps=7), 20, 0.25)
    x2 = atk2(x, y, snr)
    x1 = atk1(x, y, snr)
    d1 = (x1 - x).view(x1.size(0), -1)
    d2 = (x2 - x).view(x2.size(0), -1)
    return torch.cosine_similarity(d1, d2, dim=1)


def measure_margins(net, mlnet, x, crosslimit=10):
    atk1 = DeepFool(net, num_classes=14, max_iter=20, overshoot=0.02, refinement_steps=10, device='cuda')
    atk2 = DeepFool(mlnet, num_classes=14, max_iter=20, overshoot=0.02, refinement_steps=10, device='cuda')
    d1 = atk1(x).detach()
    d1 = torch.sqrt((d1 ** 2).sum(-2)).mean(-1)
    d2 = atk2(x).detach()
    d2 = torch.sqrt((d2 ** 2).sum(-2)).mean(-1)
    d12 = atk2.refine(x, crosslimit * d1).detach()
    d12 = torch.sqrt((d12 ** 2).sum(-2)).mean(-1)
    d21 = atk1.refine(x, crosslimit * d2).detach()
    d21 = torch.sqrt((d21 ** 2).sum(-2)).mean(-1)
    return d1, d2, d12, d21