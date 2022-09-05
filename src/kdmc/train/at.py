from kdmc.model.max_likelihood import MaxLikelihoodModel
from numpy import dtype
import torch
from tqdm import tqdm
import wandb
from kdmc.attack.core import parse_attack
from kdmc.data.core import get_num_classes
from kdmc.train.base import Trainer
import torch.nn.functional as F
import numpy as np


class ATTrainer(Trainer):

    def __init__(self, args, net, trainloader, testloader, optimizer, scheduler, sch_updt, slow_rate=5):
        super().__init__(args, net, trainloader, testloader, optimizer, scheduler, sch_updt, slow_rate)
        self.atk = parse_attack(self.net, args.atk)

    def train(self, epoch):
        print('\nEpoch: %d' % epoch)
        self.net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, batch in enumerate(tqdm(self.train_dl)):
            inputs, targets = batch['x'].to(self.device), batch['y'].to(self.device)
            snr = batch['snr'].to(self.device)
            adv_inputs = self.atk(inputs, targets, snr)
            outputs = self.net(adv_inputs)
            self.optimizer.zero_grad()
            loss = F.cross_entropy(outputs, targets)
            loss.backward()
            self.optimizer.step()
            if self.sch_updt == 'step':
                self.scheduler.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            if len(targets.shape) > 1:
                correct += predicted.eq(targets.argmax(1)).sum().item()
            else:
                correct += predicted.eq(targets).sum().item()

        wandb.log({'train.acc': 100.*correct/total, 'train.loss': train_loss/(batch_idx+1), 'epoch': epoch})


class MLATTrainer(Trainer):
    def __init__(self, args, net, trainloader, testloader, optimizer, scheduler, sch_updt, slow_rate=5):
        super().__init__(args, net, trainloader, testloader, optimizer, scheduler, sch_updt, slow_rate)
        self.atk = parse_attack(self.net, args.atk)
        self.ml_preds = self.get_ml_preds(self.train_dl, self.ml_preds)

    def train(self, epoch):
        print('\nEpoch: %d' % epoch)
        self.net.train()
        train_loss = 0
        correct = 0
        correct_ml = 0
        total = 0
        for batch_idx, batch in enumerate(tqdm(self.train_dl)):
            inputs, targets = batch['x'].to(self.device), batch['y'].to(self.device)
            idxs = batch['idx'].to(self.device, dtype=torch.long)
            ml_preds = self.ml_preds[idxs]
            snr = batch['snr'].to(self.device)
            adv_inputs = self.atk(inputs, targets, snr)
            outputs = self.net(adv_inputs)

            self.optimizer.zero_grad()
            loss = F.cross_entropy(outputs, ml_preds)
            loss.backward()
            self.optimizer.step()
            if self.sch_updt == 'step':
                self.scheduler.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets.argmax(1)).sum().item()
            correct_ml += predicted.eq(ml_preds.argmax(1)).sum().item()

        wandb.log({'train.acc': 100.*correct/total, 'train.acc_ml': 100.*correct_ml/total, 'train.loss': train_loss/(batch_idx+1), 'epoch': epoch})


class LNRATTrainer(Trainer):
    """Label noise reduced adversarial training.
    
    It skips training with inputs whose true label is different from the ML prediction at the adversarial point."""
    def __init__(self, args, net, trainloader, testloader, optimizer, scheduler, sch_updt, slow_rate=5):
        super().__init__(args, net, trainloader, testloader, optimizer, scheduler, sch_updt, slow_rate)
        self.atk = parse_attack(self.net, args.atk)
        n_signals = len(trainloader.dataset) + len(testloader.dataset)
        self.lnr_mask = torch.zeros([n_signals], device=self.device, dtype=torch.bool)

    def train(self, epoch):
        print('\nEpoch: %d' % epoch)
        self.net.train()
        train_loss = 0
        correct = 0
        correct_ml = 0
        total = 0
        for batch_idx, batch in enumerate(tqdm(self.train_dl)):
            inputs, targets = batch['x'].to(self.device), batch['y'].to(self.device)
            snr = batch['snr'].to(self.device)
            snrs = batch['snr_filt'].to(self.device)
            idxs = batch['idx'].to(self.device, dtype=torch.long)
            if epoch == 1:
                ml_preds = self.ml_model.compute_ml(inputs, snrs)
                ml_preds = self.ml_model.adapt_unsupported(ml_preds, targets)
                lnr_mask = targets.argmax(1) == ml_preds.argmax(1)
                self.lnr_mask[idxs] = lnr_mask
            else:
                lnr_mask = self.lnr_mask[idxs]
            # Filter out the inputs whose true label is different from the ML prediction
            if not lnr_mask.any():
                continue
            inputs = inputs[lnr_mask]
            targets = targets[lnr_mask]
            snr = snr[lnr_mask]
            snrs = snrs[lnr_mask]

            adv_inputs = self.atk(inputs, targets, snr)
            outputs = self.net(adv_inputs)
            ml_preds = self.ml_model.compute_advml(inputs, adv_inputs, snrs)
            ml_preds = self.ml_model.adapt_unsupported(ml_preds, targets)

            # Filter out the adversarial inputs whose true label is different
            lnr_mask = targets.argmax(1) == ml_preds.argmax(1)
            if not lnr_mask.any():
                continue
            outputs = outputs[lnr_mask]
            ml_preds = ml_preds[lnr_mask]
            targets = targets[lnr_mask]

            self.optimizer.zero_grad()
            loss = F.cross_entropy(outputs, ml_preds)
            if torch.isnan(loss):
                print('nan loss. Shape: ', loss.shape)
                continue
            loss.backward()
            self.optimizer.step()
            if self.sch_updt == 'step':
                self.scheduler.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets.argmax(1)).sum().item()
            correct_ml += predicted.eq(ml_preds.argmax(1)).sum().item()

        wandb.log({'train.acc': 100.*correct/total, 'train.acc_ml': 100.*correct_ml/total, 'train.loss': train_loss/(batch_idx+1), 'epoch': epoch})


class SelfMLATTrainer(Trainer):
    """For faster compute the ml predictions are stored. Thus, assumes there is no data augmentation"""
    def __init__(self, args, net, trainloader, testloader, optimizer, scheduler, sch_updt, slow_rate=5):
        super().__init__(args, net, trainloader, testloader, optimizer, scheduler, sch_updt, slow_rate)
        self.atk = parse_attack(self.net, args.atk)
        self.ml_preds = self.get_ml_preds(self.train_dl, self.ml_preds)
        self.alpha = args.kt_alpha
        if self.alpha is None:
            raise ValueError(f"--kt_alpha not defined")

    def train(self, epoch):
        print('\nEpoch: %d' % epoch)
        self.net.train()
        train_loss = 0
        correct = 0
        correct_ml = 0
        total = 0
        for batch_idx, batch in enumerate(tqdm(self.train_dl)):
            inputs, targets = batch['x'].to(self.device), batch['y'].to(self.device)
            idxs = batch['idx'].to(self.device, dtype=torch.long)
            ml_preds = self.ml_preds[idxs]
            snr = batch['snr'].to(self.device)
            adv_inputs = self.atk(inputs, targets, snr)
            outputs = self.net(adv_inputs)
            net_preds = F.softmax(outputs, dim=1)
            self.optimizer.zero_grad()
            loss = F.cross_entropy(outputs, ml_preds * self.alpha + net_preds * (1 - self.alpha))
            loss.backward()
            self.optimizer.step()
            if self.sch_updt == 'step':
                self.scheduler.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets.argmax(1)).sum().item()
            correct_ml += predicted.eq(ml_preds.argmax(1)).sum().item()

        wandb.log({'train.acc': 100.*correct/total, 'train.acc_ml': 100.*correct_ml/total, 'train.loss': train_loss/(batch_idx+1), 'epoch': epoch})


class SoftMLATTrainer(Trainer):
    """For faster compute the ml predictions are stored. Thus, assumes there is no data augmentation"""
    def __init__(self, args, net, trainloader, testloader, optimizer, scheduler, sch_updt, slow_rate=5):
        super().__init__(args, net, trainloader, testloader, optimizer, scheduler, sch_updt, slow_rate)
        self.atk = parse_attack(self.net, args.atk)
        self.ml_preds = self.get_ml_preds(self.train_dl, self.ml_preds)
        self.alpha = args.kt_alpha
        if self.alpha is None:
            raise ValueError(f"--kt_alpha not defined")

    def train(self, epoch):
        print('\nEpoch: %d' % epoch)
        self.net.train()
        train_loss = 0
        correct = 0
        correct_ml = 0
        total = 0
        for batch_idx, batch in enumerate(tqdm(self.train_dl)):
            inputs, targets = batch['x'].to(self.device), batch['y'].to(self.device)
            idxs = batch['idx'].to(self.device, dtype=torch.long)
            ml_preds = self.ml_preds[idxs]
            snr = batch['snr'].to(self.device)
            adv_inputs = self.atk(inputs, targets, snr)
            outputs = self.net(adv_inputs)
            self.optimizer.zero_grad()
            loss = F.cross_entropy(outputs, ml_preds * self.alpha + targets * (1 - self.alpha))
            loss.backward()
            self.optimizer.step()
            if self.sch_updt == 'step':
                self.scheduler.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets.argmax(1)).sum().item()
            correct_ml += predicted.eq(ml_preds.argmax(1)).sum().item()

        wandb.log({'train.acc': 100.*correct/total, 'train.acc_ml': 100.*correct_ml/total, 'train.loss': train_loss/(batch_idx+1), 'epoch': epoch})


class AMLATTrainer(Trainer):
    def __init__(self, args, net, trainloader, testloader, optimizer, scheduler, sch_updt, slow_rate=5):
        super().__init__(args, net, trainloader, testloader, optimizer, scheduler, sch_updt, slow_rate)
        self.atk = parse_attack(self.net, args.atk)
        self.ml_preds = self.get_adv_ml_preds(self.train_dl, self.ml_preds)

    def train(self, epoch):
        print('\nEpoch: %d' % epoch)
        self.net.train()
        train_loss = 0
        correct = 0
        correct_ml = 0
        total = 0
        for batch_idx, batch in enumerate(tqdm(self.train_dl)):
            inputs, targets = batch['x'].to(self.device), batch['y'].to(self.device)
            idxs = batch['idx'].to(self.device, dtype=torch.long)
            ml_preds = self.ml_preds[idxs]
            snr = batch['snr'].to(self.device)
            adv_inputs = self.atk(inputs, targets, snr)
            outputs = self.net(adv_inputs)

            self.optimizer.zero_grad()
            loss = F.cross_entropy(outputs, ml_preds)
            loss.backward()
            self.optimizer.step()
            if self.sch_updt == 'step':
                self.scheduler.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets.argmax(1)).sum().item()
            correct_ml += predicted.eq(ml_preds.argmax(1)).sum().item()

        wandb.log({'train.acc': 100.*correct/total, 'train.acc_ml': 100.*correct_ml/total, 'train.loss': train_loss/(batch_idx+1), 'epoch': epoch})


class MMLATTrainer(Trainer):
    """For faster compute the ml predictions are stored. Thus, assumes there is no data augmentation. 
    It does not assume every training sample has a 'x_ml' attribute."""
    def __init__(self, args, net, trainloader, testloader, optimizer, scheduler, sch_updt, slow_rate=5):
        super().__init__(args, net, trainloader, testloader, optimizer, scheduler, sch_updt, slow_rate)
        self.ml_preds, self.is_ml = self.get_ml_preds(self.train_dl, self.ml_preds)
    
    def __init__(self, args, net, trainloader, testloader, optimizer, scheduler, sch_updt, slow_rate=5):
        super().__init__(args, net, trainloader, testloader, optimizer, scheduler, sch_updt, slow_rate)
        self.atk = parse_attack(self.net, args.atk)
        self.ml_preds, self.is_ml = self.get_ml_preds(self.train_dl, self.ml_preds)

    def train(self, epoch):
        print('\nEpoch: %d' % epoch)
        self.net.train()
        train_loss = 0
        correct = 0
        correct_ml = 0
        total = 0
        total_ml = 0
        for batch_idx, batch in enumerate(tqdm(self.train_dl)):
            inputs, targets = batch['x'].to(self.device), batch['y'].to(self.device)
            idxs = batch['idx'].to(self.device, dtype=torch.long)
            is_ml = self.is_ml[idxs].sum() > 0
            if is_ml:
                ml_preds = self.ml_preds[idxs]
            else:
                ml_preds = targets
            snr = batch['snr'].to(self.device)
            adv_inputs = self.atk(inputs, targets, snr)
            outputs = self.net(adv_inputs)

            self.optimizer.zero_grad()
            loss = F.cross_entropy(outputs, ml_preds)
            loss.backward()
            self.optimizer.step()
            if self.sch_updt == 'step':
                self.scheduler.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets.argmax(1)).sum().item()
            if is_ml:
                correct_ml += predicted.eq(ml_preds.argmax(1)).sum().item()
                total_ml += targets.size(0)

        wandb.log({'train.acc': 100.*correct/total, 'train.acc_ml': 100.*correct_ml/total_ml, 'train.loss': train_loss/(batch_idx+1), 'epoch': epoch})
    
    def get_ml_preds(self, dl, ml_preds):
        is_ml = torch.ones([ml_preds.shape[0]])
        states_path = self.data_path.joinpath('synthetic/signal/states')
        ml_model = MaxLikelihoodModel(states_path, self.classes, device=self.device)
        print('Computing ML predictions')
        for batch in tqdm(dl):
            idx = batch['idx'].to(self.device, dtype=torch.long)
            # The added condition that makes it able to process mixed data
            if not hasattr(batch, 'sps'):
                is_ml[idx] = 0
                continue
            sps = batch['sps']
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
        return ml_preds, is_ml