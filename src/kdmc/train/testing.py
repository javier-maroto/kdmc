from functools import partial
import torch
from tqdm import tqdm
import wandb

from kdmc.attack.core import SPR_Attack
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
        }, commit=False)


def get_cosine_similarity_ml(trainer):
    """
    Compute the cosine similarity between the model and the ML model adversarial attacks.
    """
    res = []

    def new_forward(model, x):
        return model(x)[:14]
    model = trainer.net
    model.forward = partial(new_forward, model)
    ml_model = trainer.ml_model.return_ml_model()

    atk = SPR_Attack(PGD(model, steps=7), 20, 0.25)
    for batch in tqdm(trainer.test_dl):
        targets = batch['y'].to(trainer.device)
        mask = targets.argmax(dim=-1) < 14
        print(mask.sum())
        targets = targets[mask, :14]
        print(targets.shape)
        if targets.shape[0] == 0:
            print('escape')
            continue
        inputs = batch['x'].to(trainer.device)[mask]
        snr = batch['snr'].to(trainer.device)[mask]
        snr_ml = batch['snr_filt'].to(trainer.device)[mask]
        print(inputs.shape)
        ml_model.snr = snr_ml
        atk_ml = SPR_Attack(PGD(ml_model, steps=7), 20, 0.25)
        x_adv = atk(inputs, targets, snr)
        print(x_adv.shape)
        x_adv_ml = atk_ml(inputs, targets, snr)
        print(x_adv_ml.shape)
        d_model = (x_adv - inputs).view(x_adv.size(0), -1)
        d_ml = (x_adv_ml - inputs).view(x_adv_ml.size(0), -1)
        print(d_ml.shape)
        cs_ml = torch.cosine_similarity(d_model, d_ml, dim=1).cpu().numpy()
        res.extend(cs_ml)
        print(res)
    return res
    