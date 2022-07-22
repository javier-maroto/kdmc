import abc
from copy import deepcopy
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
from kdmc.data.core import get_classes, get_num_classes
from kdmc.model.max_likelihood import MaxLikelihoodModel
from kdmc.plot import plot_acc_vs_snr
from kdmc.train.testing import get_acc_metrics, get_acc_metrics_wo_ml, measure_cosine_similarity, measure_margins


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
        self.root_path = args.root_path
        self.data_path = args.data_path
        self.classes = get_classes(self.dataset)
        
        self.ml_preds = None
        if self.dataset not in ('rml2016.10a'):
            self.ml_preds = torch.zeros(
                [len(self.train_dl.dataset) + len(self.test_dl.dataset), get_num_classes(self.dataset)],
                device=self.device)
            self.ml_preds = self.get_ml_preds(self.test_dl, self.ml_preds)
            
        # Gradient clipping
        if args.grad_clip is not None:
            for p in self.net.parameters():
                p.register_hook(lambda grad: torch.clamp(grad, -args.grad_clip, args.grad_clip))

    def get_ml_preds(self, dl, ml_preds):
        states_path = self.data_path.joinpath('synthetic/signal/states')
        ml_model = MaxLikelihoodModel(states_path, self.classes, device=self.device)
        print('Computing ML predictions')
        for batch in tqdm(dl):
            sps = batch['sps']
            idx = batch['idx'].to(self.device, dtype=torch.long)
            ml_preds[idx] = batch['y'].to(self.device)
            target_mask = (batch['y'] * ml_model.supported).sum(-1) == 1
            for a in sps.unique():
                mask = ((sps == a) & target_mask).to(self.device)
                if mask.sum() == 0:
                    continue
                x = batch['x_ml'][mask].to(self.device)
                snr_ml = batch['snr_filt'][mask].to(self.device)
                y = batch['y'][mask].to(self.device)
                idx = batch['idx'][mask].to(self.device, dtype=torch.long)
                ls = a.numpy().astype(np.int)
                ml_preds_ = ml_model.compute_ml_symb(x, snr_ml, sps=ls)
                ml_preds_ = ml_model.adapt_unsupported(ml_preds_, y)
                ml_preds[idx] = ml_preds_
        return ml_preds

    def get_adv_ml_preds(self, batch, x_adv):
        ml_model = MaxLikelihoodModel(
            self.data_path.joinpath('synthetic/signal/states'), self.classes, device=self.device)
        sps = batch['sps']
        rolloff = batch['rolloff']
        n_span_symb = 8
        target_mask = (batch['y'] * ml_model.supported).sum(-1) == 1
        ml_preds = torch.zeros([x_adv.shape[0], get_num_classes(self.dataset)], device=self.device)
        for a in sps.unique():
            for b in rolloff.unique():
                mask = ((sps == a) & (rolloff == b) & target_mask).to(self.device)
                if mask.sum() == 0:
                    continue
                x = batch['x'][mask].to(self.device)
                snr_ml = batch['snr_filt'][mask].to(self.device)
                y = batch['y'][mask].to(self.device)
                ls = a.numpy().astype(np.int)
                rf = b.numpy()
                while ls * n_span_symb >= x.shape[-1]:
                    n_span_symb //= 2
                ml_model.create_filter(Ls=ls, rolloff=rf, n_span_symb=n_span_symb)
                ml_preds_ = ml_model.compute_advml(x, x_adv[mask], snr_ml)
                ml_preds_ = ml_model.adapt_unsupported(ml_preds_, y)
                ml_preds[mask] = ml_preds_
        y = batch['y'][~target_mask].to(self.device)
        ml_preds[~target_mask] = y
        return ml_preds

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
                idx = batch['idx'].to(self.device, dtype=torch.long)
                outputs = self.net(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct['clean'] += predicted.eq(targets.argmax(1)).sum().item()
                # Compute ML accuracy
                if self.ml_preds is not None:
                    ml_preds = self.ml_preds[idx]
                    correct['clean_ml'] += predicted.eq(ml_preds.argmax(1)).sum().item()
        if self.ml_preds is not None:
            wandb.log({'test.acc_ml': 100.*correct['clean_ml']/total}, commit=False)
        wandb.log({
            'test.acc': 100.*correct['clean']/total,
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

    def test(self):
        res_path = os.path.join("results", self.id)
        if not os.path.exists(res_path):
            os.makedirs(res_path)
        
        self.net.eval()
        res = {'true': [], 'clean': [], 'snr': []}
        if self.ml_preds is not None:
            res['clean_ml'] = []
        total = 0
        atks = self.get_test_attacks()
        for key in atks:
            res[key] = []
            if self.ml_preds is not None:
                res[f'{key}_ml'] = []
        #res.update(self.get_geometric_metrics())
        with torch.no_grad():
            for batch in tqdm(self.test_dl):
                inputs, targets = batch['x'].to(self.device), batch['y'].to(self.device)
                idx = batch['idx'].to(self.device, dtype=torch.long)
                snr = batch['snr'].to(self.device)
                outputs = self.net(inputs)
                predicted = outputs.argmax(1)
                total += targets.size(0)
                res['true'].extend(batch['y'].argmax(1).numpy())
                res['clean'].extend(predicted.cpu().numpy())
                res['snr'].extend(batch['snr'].numpy())
                if self.ml_preds is not None:
                    ml_preds = self.ml_preds[idx]
                    res['clean_ml'].extend(ml_preds.argmax(1).cpu().numpy())

                for key, atk in atks.items():
                    with torch.enable_grad():
                        x_adv = atk(inputs, targets, snr)
                    outputs = self.net(x_adv)
                    predicted = outputs.argmax(1)
                    res[key].extend(predicted.cpu().numpy())
                    if self.ml_preds is not None:
                        ml_preds = self.get_adv_ml_preds(batch, x_adv)
                        res[f'{key}_ml'].extend(ml_preds.argmax(1).cpu().numpy())
        for key, value in res.items():
            res[key] = np.array(value)
        df = pd.DataFrame(res)
        if self.ml_preds is not None:
            gam = get_acc_metrics
        else:
            gam = get_acc_metrics_wo_ml
        gam(df)
        for key in atks:
            gam(df, key)
            """
        wandb.log({
            'cosine_similarity': df['cs_ml'].mean(), 
            'margin_nn': df['df_nn'].mean(),
            'margin_ml': df['df_ml'].mean(),
            'margin_nn_towards_ml_margin': df['df_nn_ml'].mean(),
            'margin_ml_towards_nn_margin': df['df_ml_nn'].mean()})"""
        return res
    
    def get_geometric_metrics(self):
        model, ml_model = self.create_reduced_models()
        res = {'cs_ml': [], 'df_nn': [], 'df_ml': [], 'df_nn_ml': [], 'df_ml_nn': []}

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
            cs_ml = measure_cosine_similarity(model, ml_model, inputs, targets, snr)
            d1, d2, d12, d21 = measure_margins(model, ml_model, inputs)
            res['cs_ml'].extend(cs_ml.cpu().numpy())
            res['df_nn'].extend(d1.cpu().numpy())
            res['df_ml'].extend(d2.cpu().numpy())
            res['df_nn_ml'].extend(d12.cpu().numpy())
            res['df_ml_nn'].extend(d21.cpu().numpy())
        return res
    
    def create_reduced_models(self):
        """
        Create a reduced model that takes only the first 14 classes, and the ML model.
        """
        class ReducedModel(nn.Module):
            def __init__(self, model):
                super(ReducedModel, self).__init__()
                self.model = model

            def forward(self, x):
                return self.model(x)[:, :14]
                
        model = ReducedModel(self.net)
        ml_model = self.ml_model.return_ml_model()
        return model, ml_model

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
        base_path = self.root_path.joinpath(
            f'checkpoint/{self.dataset}/{self.arch}/{self.id}/{self.seed}')
        if not os.path.isdir(base_path):
            os.makedirs(base_path, exist_ok=True)
        if self.save_inter:
            torch.save(state, base_path.joinpath(f'{epoch}.pth'))
        else:
            torch.save(state, base_path.joinpath(f'ckpt_last.pth'))

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


class MLTrainer(Trainer):

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
        self.root_path = args.root_path
        self.data_path = args.data_path
        self.classes = get_classes(self.dataset)
        
        self.ml_preds = None
        if self.dataset not in ('rml2016.10a'):
            self.ml_preds = torch.zeros(
                [len(self.train_dl.dataset) + len(self.test_dl.dataset), get_num_classes(self.dataset)],
                device=self.device)
            self.ml_preds = self.get_ml_preds(self.test_dl, self.ml_preds)

    def train(self, epoch):
        pass

    def test(self):
        res_path = os.path.join("results", self.id)
        if not os.path.exists(res_path):
            os.makedirs(res_path)
        ml = MaxLikelihoodModel(
            self.data_path.joinpath('synthetic/signal/states'), self.classes, device=self.device)
        res = {'true': [], 'clean': [], 'pgd-7_20dB': [], 'snr': []}
        if self.ml_preds is not None:
            res['clean_ml'] = []
        total = 0
        #res.update(self.get_geometric_metrics())
        with torch.no_grad():
            for batch in tqdm(self.test_dl):
                targets = batch['y'].to(self.device)
                idx = batch['idx'].to(self.device, dtype=torch.long)
                outputs = self.ml_preds[idx]
                predicted = outputs.argmax(1)
                total += targets.size(0)
                res['true'].extend(batch['y'].argmax(1).numpy())
                res['clean'].extend(predicted.cpu().numpy())
                res['snr'].extend(batch['snr'].numpy())
                if self.ml_preds is not None:
                    ml_preds = self.ml_preds[idx]
                    res['clean_ml'].extend(ml_preds.argmax(1).cpu().numpy())
                snr_ml = batch['snr_filt'].to(self.device)
                ml_model = ml.return_ml_model(snr_ml)
                atk = SPR_Attack(PGD(ml_model, steps=7), 20, 0.25)
                x = batch['x_ml'].to(self.device)
                x = x[..., :(x.shape[-1] // 8)] # TODO: sps can change with different datasets
                with torch.enable_grad():
                    x_adv = atk(x, targets[..., :14], snr_ml)
                new_snr = -20 * torch.log10(10**(-snr_ml/20) + 10**(-20/20))
                ml_preds = ml.compute_ml_symb(x_adv, new_snr)  
                ml_preds = ml.adapt_unsupported(ml_preds, targets)
                predicted = ml_preds.argmax(1)
                res['pgd-7_20dB'].extend(predicted.cpu().numpy())

        for key, value in res.items():
            res[key] = np.array(value)
        df = pd.DataFrame(res)
        get_acc_metrics(df)
        get_acc_metrics_wo_ml(df, atk_key='pgd-7_20dB')
        return res
    