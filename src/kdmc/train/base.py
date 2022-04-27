import abc
from copy import deepcopy
import os
import pandas as pd
import numpy as np

import torch
from tqdm import tqdm
import wandb
import torch.nn.functional as F
from kdmc.attack.core import SPR_Attack

from kdmc.attack.pgd import PGD
from kdmc.data.core import get_num_classes
from kdmc.plot import plot_acc_vs_snr


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
        test_loss = {
            'clean': 0
        }
        correct = {
            'clean': 0
        }
        total = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.test_dl)):
                inputs, targets = batch['x'].to(self.device), batch['y'].to(self.device)
                outputs = self.net(inputs)
                loss = F.cross_entropy(outputs, targets)
                test_loss['clean'] += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                if len(targets.shape) > 1:
                    correct['clean'] += predicted.eq(targets.argmax(1)).sum().item()
                else:
                    correct['clean'] += predicted.eq(targets).sum().item()

        wandb.log({
            'test.acc': 100.*correct['clean']/total, 'test.loss': test_loss['clean']/(batch_idx+1),
            'lr': self.scheduler.get_last_lr()[0], 'epoch': epoch})

    def val_slow(self, epoch):
        self.net.eval()
        test_loss = {
            'clean': 0
        }
        correct = {
            'clean': 0
        }
        total = 0
        slow_atks = self.get_val_attacks()
        for key in slow_atks:
            test_loss[key] = 0
            correct[key] = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.test_dl)):
                inputs, targets = batch['x'].to(self.device), batch['y'].to(self.device)
                outputs = self.net(inputs)
                loss = F.cross_entropy(outputs, targets)
                test_loss['clean'] += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                if len(targets.shape) > 1:
                    correct['clean'] += predicted.eq(targets.argmax(1)).sum().item()
                else:
                    correct['clean'] += predicted.eq(targets).sum().item()

                for key, atk in slow_atks.items():
                    with torch.enable_grad():
                        x_adv = atk(inputs, targets)
                    outputs = self.net(x_adv)
                    loss = F.cross_entropy(outputs, targets)
                    test_loss[key] += loss.item()
                    _, predicted = outputs.max(1)
                    if len(targets.shape) > 1:
                        correct[key] += predicted.eq(targets.argmax(1)).sum().item()
                    else:
                        correct[key] += predicted.eq(targets).sum().item()

        log_dict = {f'test.{key}_acc': 100.*correct[key]/total for key in slow_atks}
        log_dict.update({f'test.{key}_loss': test_loss[key]/(batch_idx+1) for key in slow_atks})
        wandb.log(log_dict, commit=False)
        wandb.log({
            'test.acc': 100.*correct['clean']/total, 'test.loss': test_loss['clean']/(batch_idx+1),
            'lr': self.scheduler.get_last_lr()[0], 'epoch': epoch})

    def test(self):
        res_path = os.path.join("results", self.id)
        if not os.path.exists(res_path):
            os.makedirs(res_path)
        
        self.net.eval()
        test_loss = {'clean': 0}
        res = {'true': [], 'clean': [], 'snr': []}
        total = 0
        slow_atks = self.get_test_attacks()
        for key in slow_atks:
            test_loss[key] = 0
            res[key] = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.test_dl)):
                inputs, targets = batch['x'].to(self.device), batch['y'].to(self.device)
                outputs = self.net(inputs)
                loss = F.cross_entropy(outputs, targets)
                test_loss['clean'] += loss.item()
                predicted = outputs.argmax(1)
                total += targets.size(0)
                if len(targets.shape) > 1:
                    res['true'].extend(batch['y'].argmax(1).numpy())
                else:
                    res['true'].extend(batch['y'].numpy())
                res['clean'].extend(predicted.cpu().numpy())
                res['snr'].extend(batch['snr'].numpy())

                for key, atk in slow_atks.items():
                    with torch.enable_grad():
                        x_adv = atk(inputs, targets)
                    outputs = self.net(x_adv)
                    loss = F.cross_entropy(outputs, targets)
                    test_loss[key] += loss.item()
                    _, predicted = outputs.max(1)
                    res[key].extend(predicted.cpu().numpy())
        for key, value in res.items():
            res[key] = np.array(value)
        df = pd.DataFrame(res)
        df['acc'] = df['true']==df['clean']
        df_snr = df.groupby('snr', as_index=False)['acc'].mean().sort_values('snr')
        fig = plot_acc_vs_snr(df_snr.acc, df_snr.snr, title="Accuracy vs SNR")
        fig.savefig(os.path.join(res_path, "acc_vs_snr.png"))
        log_dict = {f'test.{key}_acc': 100.*(df['true']==df[key]).mean() for key in slow_atks}
        log_dict.update({f'test.{key}_loss': test_loss[key]/(batch_idx+1) for key in slow_atks})
        wandb.log(log_dict, commit=False)
        wandb.log({
            'test.acc': 100.*df.acc.mean(), 'test.loss': test_loss['clean']/(batch_idx+1)})
    
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