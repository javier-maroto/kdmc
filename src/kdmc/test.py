import logging
import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import wandb
import abc

from kdmc.attack.core import SPR_Attack
from kdmc.attack.pgd import PGD
from kdmc.utils import compute_sample_energy


class Tester(abc.ABC):
    def __init__(self, args, net, classes=None):
        self.net = net
        self.device = args.device
        self.id = args.id
        self.classes = classes

    @abc.abstractmethod
    def test(self, test_dl, ml_preds=None, epoch=None):
        pass


class FastValTester(Tester):

    def __init__(self, net, scheduler):
        super().__init__(net)
        self.scheduler = scheduler

    def test(self, test_dl, ml_preds=None, epoch=None):
        self.net.eval()
        correct = {
            'clean': 0,
            'clean_ml': 0,
        }
        total = 0
        with torch.no_grad():
            for batch in tqdm(test_dl):
                inputs, targets = batch['x'].to(self.device), batch['y'].to(self.device)
                idx = batch['idx'].to(self.device, dtype=torch.long)
                outputs = self.net(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct['clean'] += predicted.eq(targets.argmax(1)).sum().item()
                # Compute ML accuracy
                if ml_preds is not None:
                    ml_preds = ml_preds[idx]
                    correct['clean_ml'] += predicted.eq(ml_preds.argmax(1)).sum().item()
        if ml_preds is not None:
            wandb.log({'test.acc_ml': 100.*correct['clean_ml']/total}, commit=False)
        if epoch is not None:
            wandb.log({'epoch': epoch}, commit=False)
        wandb.log({
            'test.acc': 100.*correct['clean']/total,
            'lr': self.scheduler.get_last_lr()[0]})


class SlowValTester(Tester):
    def __init__(self, net, scheduler):
        super().__init__(net)
        self.scheduler = scheduler

    def test(self, test_dl, ml_preds=None, epoch=None):
        self.net.eval()
        correct = {
            'clean': 0,
        }
        ml_correct = {
            'clean': 0,
        }
        total = 0
        slow_atks = {
            'pgd-7_20dB': SPR_Attack(PGD(self.net, steps=7), 20, 0.25)
        }
        for key in slow_atks:
            correct[key] = 0
            ml_correct[key] = 0
        with torch.no_grad():
            for batch in tqdm(test_dl):
                inputs, targets = batch['x'].to(self.device), batch['y'].to(self.device)
                idx = batch['idx'].to(self.device, dtype=torch.long)
                snr = batch['snr'].to(self.device)
                outputs = self.net(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct['clean'] += predicted.eq(targets.argmax(1)).sum().item()
                # Compute ML accuracy
                if self.ml_preds is not None:
                    ml_preds = self.ml_preds[idx]
                    ml_correct['clean'] += predicted.eq(ml_preds.argmax(1)).sum().item()

                for key, atk in slow_atks.items():
                    with torch.enable_grad():
                        x_adv = atk(inputs, targets, snr)
                    outputs = self.net(x_adv)
                    _, predicted = outputs.max(1)
                    correct[key] += predicted.eq(targets.argmax(1)).sum().item()
                    # Compute ML accuracy
                    if self.ml_preds is not None:
                        ml_preds = self.get_adv_ml_preds(batch, x_adv)
                        ml_correct[key] += predicted.eq(ml_preds.argmax(1)).sum().item()

        log_dict = {f'test.{key}_acc': 100.*correct[key]/total for key in slow_atks}
        if self.ml_preds is not None:
            log_dict.update({f'test.{key}_acc_ml': 100.*ml_correct[key]/total for key in slow_atks})
        wandb.log(log_dict, commit=False)
        if self.ml_preds is not None:
            wandb.log({'test.acc_ml': 100.*ml_correct['clean']/total}, commit=False)
        wandb.log({
            'test.acc': 100.*correct['clean']/total,
            'lr': self.scheduler.get_last_lr()[0], 'epoch': epoch})


class FullTester(Tester):

    def test(self, test_dl):
        res_path = os.path.join("results", self.id)
        if not os.path.exists(res_path):
            os.makedirs(res_path)
        
        self.net.eval()
        res = {'true': [], 'clean': [], 'snr': []}
        total = 0
        atks = {
            'pgd-7_20dB': SPR_Attack(PGD(self.net, steps=7), 20, 0.25)
        }
        for key in atks:
            res[key] = []
        with torch.no_grad():
            for batch in tqdm(test_dl):
                inputs, targets = batch['x'].to(self.device), batch['y'].to(self.device)
                snr = batch['snr'].to(self.device)
                outputs = self.net(inputs)
                predicted = outputs.argmax(1)
                total += targets.size(0)
                res['true'].extend(batch['y'].argmax(1).numpy())
                res['clean'].extend(predicted.cpu().numpy())
                res['snr'].extend(batch['snr'].numpy())

                for key, atk in atks.items():
                    with torch.enable_grad():
                        x_adv = atk(inputs, targets, snr)
                    outputs = self.net(x_adv)
                    predicted = outputs.argmax(1)
                    res[key].extend(predicted.cpu().numpy())
            logging.debug(f"Inputs: {inputs.shape}")
            logging.debug(compute_sample_energy(inputs))
            logging.debug(inputs)
            logging.debug(f"Outputs: {outputs.shape}")
            logging.debug(outputs)

        for key, value in res.items():
            res[key] = np.array(value)
        df = pd.DataFrame(res)
        self.get_acc_metrics(df)
        for key in atks:
            self.get_acc_metrics(df, key)
        return res

    def get_acc_metrics(self, df, atk_key=None):
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
            f'test/{label}conf_matrix': wandb.plot.confusion_matrix(
                y_true=df['true'], preds=df[col], class_names=self.classes, title="Confusion matrix")
        })
