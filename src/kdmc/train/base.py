import abc
from copy import deepcopy
from functools import partial
import os
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from tqdm import tqdm
import wandb
import torch.nn.functional as F
from kdmc.attack.core import SPR_Attack

from kdmc.attack.pgd import PGD
from kdmc.model.max_likelihood import MaxLikelihoodModel
from kdmc.plot import plot_acc_vs_snr
from kdmc.train.testing import get_acc_metrics, get_cosine_similarity_ml


class Trainer(abc.ABC):

    def __init__(self, args, net, trainloader, testloader, optimizer, scheduler, sch_updt, slow_rate=5):
        self.id = args.id
        self.seed = args.seed
        self.arch = args.arch
        self.dataset = args.dataset
        self.net = net
        self.train_dl = trainloader
        self.test_dl = testloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.sch_updt = sch_updt
        self.slow_rate = slow_rate
        self.device = args.device
        self.save_inter = args.save_inter
        self.ml_model = MaxLikelihoodModel(
            f'{args.root_path}/data/synthetic/signal/states', device=self.device)
        self.ml_model.create_filter()

    def loop_fast(self, epoch):
        self.train(epoch)
        if self.sch_updt == 'epoch':
            self.scheduler.step()
        if epoch % self.slow_rate == 0:
            self.save_ckpt(epoch)

    def loop(self, epoch):
        self.train(epoch)
        if self.sch_updt == 'epoch':
            self.scheduler.step()
        if epoch % self.slow_rate == 0:
            self.val_slow(epoch)
            self.save_ckpt(epoch)
        else:
            self.val_fast(epoch)

    @abc.abstractmethod
    def train(self, epoch):
        pass

    def val_fast(self, epoch):
        self.net.eval()
        correct = {
            'clean': 0,
            'clean_ml': 0,
        }
        total = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.test_dl)):
                inputs, targets = batch['x'].to(self.device), batch['y'].to(self.device)
                outputs = self.net(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct['clean'] += predicted.eq(targets.argmax(1)).sum().item()
                # Compute ML accuracy
                ml_preds = self.ml_model.compute_ml(inputs, batch['snr_filt'].to(self.device))
                correct['clean_ml'] += predicted.eq(ml_preds.argmax(1)).sum().item()

        wandb.log({
            'test.acc': 100.*correct['clean']/total, 'test.acc_ml': 100.*correct['clean_ml']/total,
            'lr': self.scheduler.get_last_lr()[0], 'epoch': epoch})

    def val_slow(self, epoch):
        self.net.eval()
        correct = {
            'clean': 0,
        }
        ml_correct = {
            'clean': 0,
        }
        total = 0
        slow_atks = self.get_val_attacks()
        for key in slow_atks:
            correct[key] = 0
            ml_correct[key] = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.test_dl)):
                inputs, targets = batch['x'].to(self.device), batch['y'].to(self.device)
                snr = batch['snr'].to(self.device)
                snr_ml = batch['snr_filt'].to(self.device)
                outputs = self.net(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct['clean'] += predicted.eq(targets.argmax(1)).sum().item()
                # Compute ML accuracy
                ml_preds = self.ml_model.compute_ml(inputs, snr_ml)
                ml_correct['clean'] += predicted.eq(ml_preds.argmax(1)).sum().item()

                for key, atk in slow_atks.items():
                    with torch.enable_grad():
                        x_adv = atk(inputs, targets, snr)
                    outputs = self.net(x_adv)
                    _, predicted = outputs.max(1)
                    correct[key] += predicted.eq(targets.argmax(1)).sum().item()
                    # Compute ML accuracy
                    ml_preds = self.ml_model.compute_advml(inputs, x_adv, snr_ml)
                    ml_correct[key] += predicted.eq(ml_preds.argmax(1)).sum().item()

        log_dict = {f'test.{key}_acc': 100.*correct[key]/total for key in slow_atks}
        log_dict.update({f'test.{key}_acc_ml': 100.*ml_correct[key]/total for key in slow_atks})
        wandb.log(log_dict, commit=False)
        wandb.log({
            'test.acc': 100.*correct['clean']/total, 'test.acc_ml': 100.*ml_correct['clean']/total,
            'lr': self.scheduler.get_last_lr()[0], 'epoch': epoch})

    def test(self):
        res_path = os.path.join("results", self.id)
        if not os.path.exists(res_path):
            os.makedirs(res_path)
        
        self.net.eval()
        res = {'true': [], 'clean': [], 'clean_ml': [], 'snr': []}
        total = 0
        atks = self.get_test_attacks()
        for key in atks:
            res[key] = []
            res[f'{key}_ml'] = []
        res['cs_ml'] = self.get_cosine_similarity_ml()
        with torch.no_grad():
            for batch in tqdm(self.test_dl):
                inputs, targets = batch['x'].to(self.device), batch['y'].to(self.device)
                snr = batch['snr'].to(self.device)
                snr_ml = batch['snr_filt'].to(self.device)
                outputs = self.net(inputs)
                predicted = outputs.argmax(1)
                ml_preds = self.ml_model.compute_ml(inputs, snr_ml)
                total += targets.size(0)
                res['true'].extend(batch['y'].argmax(1).numpy())
                res['clean'].extend(predicted.cpu().numpy())
                res['clean_ml'].extend(ml_preds.argmax(1).cpu().numpy())
                res['snr'].extend(batch['snr'].numpy())

                for key, atk in atks.items():
                    with torch.enable_grad():
                        x_adv = atk(inputs, targets, snr)
                    outputs = self.net(x_adv)
                    predicted = outputs.argmax(1)
                    ml_preds = self.ml_model.compute_advml(inputs, x_adv, snr_ml)
                    res[key].extend(predicted.cpu().numpy())
                    res[f'{key}_ml'].extend(ml_preds.argmax(1).cpu().numpy())
        for key, value in res.items():
            res[key] = np.array(value)
        df = pd.DataFrame(res)
        get_acc_metrics(df)
        for key in atks:
            get_acc_metrics(df, key)
        wandb.log({'cosine_similarity': df['cs_ml'].mean()})
    
    def get_cosine_similarity_ml(self):
        """
        Compute the cosine similarity between the model and the ML model adversarial attacks.
        """
        res = []

        class ReducedModel(nn.Module):
            def __init__(self, model):
                super(ReducedModel, self).__init__()
                self.model = model

            def forward(self, x):
                return self.model(x)[:, :14]
        
        model = ReducedModel(self.net)
        ml_model = self.ml_model.return_ml_model()

        atk = SPR_Attack(PGD(model, steps=7), 20, 0.25)
        for batch in tqdm(self.test_dl):
            targets = batch['y'].to(self.device)
            mask = targets.argmax(dim=-1) < 14
            targets = targets[mask, :14]
            if targets.shape[0] == 0:
                continue
            inputs = batch['x'].to(self.device)[mask]
            snr = batch['snr'].to(self.device)[mask]
            snr_ml = batch['snr_filt'].to(self.device)[mask]
            ml_model.snr = snr_ml
            atk_ml = SPR_Attack(PGD(ml_model, steps=7), 20, 0.25)
            x_adv = atk(inputs, targets, snr)
            x_adv_ml = atk_ml(inputs, targets, snr)
            d_model = (x_adv - inputs).view(x_adv.size(0), -1)
            d_ml = (x_adv_ml - inputs).view(x_adv_ml.size(0), -1)
            cs_ml = torch.cosine_similarity(d_model, d_ml, dim=1).cpu().numpy()
            res.extend(cs_ml)
        return res
    
    def save_ckpt(self, epoch):
        # Save checkpoint.
        print('Saving..')
        state = {
            'arch': self.arch,
            'dataset': self.dataset,
            'net': self.net.state_dict(),
            'optim': self.optimizer.state_dict(),
            'sched': self.scheduler.state_dict(),
            'epoch': epoch,
        }
        base_path = f'checkpoint/{self.dataset}/{self.arch}/{self.id}/{self.seed}'
        if not os.path.isdir(base_path):
            os.makedirs(base_path, exist_ok=True)
        if self.save_inter:
            torch.save(state, f'{base_path}/{epoch}.pth')
        else:
            torch.save(state, f'{base_path}/ckpt_last.pth')

    def get_val_attacks(self):
        return {
            'pgd-7_20dB': SPR_Attack(PGD(self.net, steps=7), 20, 0.25),  # Not clamped
        }
        
    def get_test_attacks(self):
        return {
            'pgd-7_20dB': SPR_Attack(PGD(self.net, steps=7), 20, 0.25)
        }


class KTTrainer(Trainer):
    def __init__(self, args, net, trainloader, testloader, optimizer, scheduler, sch_updt, slow_rate=5):
        super().__init__(args, net, trainloader, testloader, optimizer, scheduler, sch_updt, slow_rate=slow_rate)
        n_kt = len(args.kt_path)
        if len(args.ens_epoch) == 0:
            args.ens_epoch = [-1 for _ in range(n_kt)]
        elif len(args.ens_epoch) == 1:
            args.ens_epoch = [args.ens_epoch[0] for _ in range(n_kt)]
        self.net_kt = []
        for kt_path, ens_epoch in zip(args.kt_path, args.ens_epoch):
            self.net_kt.append(self.load_kt_model(args, kt_path, ens_epoch))
        self.alpha = args.kt_alpha
        if len(args.kt_beta) == 0:
            self.beta = [1./n_kt for _ in range(n_kt)]
        else:
            self.beta = args.kt_beta

    @classmethod
    def load_kt_model(cls, args, kt_path, ens_epoch):
        from kdmc.train.core import create_model
        if kt_path.startswith('robustbench'):
            from robustbench.utils import load_model
            net = load_model(model_name=kt_path.split('/')[1], dataset='cifar10', threat_model='Linf', model_dir=f'{args.root_path}/models/robustbench/')
            net.to(args.device)
            net.eval()
            return net
        if ens_epoch == -1:
            checkpoint = torch.load(f'{args.root_path}/checkpoint/{kt_path}/ckpt_last.pth')
        else:
            checkpoint = torch.load(f'{args.root_path}/checkpoint/{kt_path}/{ens_epoch}.pth')
        kt_args = deepcopy(args)
        kt_args.arch = checkpoint['arch']
        kt_args.dataset = checkpoint['dataset']
        net = create_model(kt_args)  # It may break if kt dataset is different and has extra args
        net.load_state_dict(checkpoint['net'])
        net.eval()
        return net
    
    def logits_kt(self, inputs):
        return [net(inputs) for net in self.net_kt]

    def pred_kt(self, inputs):
        # Ensemble case: it probably is better to average predictions than logits
        logits = self.logits_kt(inputs)
        preds = [F.softmax(l, dim=-1) * b for l, b in zip(logits, self.beta)]
        return torch.sum(torch.stack(preds), dim=0)
