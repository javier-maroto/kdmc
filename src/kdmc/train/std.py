import torch
from tqdm import tqdm
import wandb
from kdmc.data.core import get_num_classes
from kdmc.train.base import Trainer
import torch.nn.functional as F


class STDTrainer(Trainer):
    def train(self, epoch, profiler=None):
        print('\nEpoch: %d' % epoch)
        self.net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, batch in enumerate(tqdm(self.train_dl)):
            inputs, targets = batch['x'].to(self.device), batch['y'].to(self.device)
            outputs = self.net(inputs)
            self.optimizer.zero_grad()
            loss = F.cross_entropy(outputs, targets)
            loss.backward()
            self.optimizer.step()
            if self.sch_updt == 'step':
                self.scheduler.step()
            if profiler is not None:
                profiler.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets.argmax(1)).sum().item()

        wandb.log({'train.acc': 100.*correct/total, 'train.loss': train_loss/(batch_idx+1), 'epoch': epoch})


class MLSTDTrainer(Trainer):
    """For faster compute the ml predictions are stored. Thus, assumes there is no data augmentation"""
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
            outputs = self.net(inputs)
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


class MLSTDTrainer(Trainer):
    """For faster compute the ml predictions are stored. Thus, assumes there is no data augmentation"""
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
            outputs = self.net(inputs)
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


class LNRSTDTrainer(Trainer):
    """Label noise reduced standard training.
    
    It skips training with inputs whose true label is different from the ML prediction."""
    def __init__(self, args, net, trainloader, testloader, optimizer, scheduler, sch_updt, slow_rate=5):
        super().__init__(args, net, trainloader, testloader, optimizer, scheduler, sch_updt, slow_rate)
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
            snrs = batch['snr_filt'].to(self.device)
            idxs = batch['idx'].to(self.device, dtype=torch.long)
            ml_preds = self.ml_preds[idxs]
            if epoch == 1:
                self.lnr_mask[idxs] = targets.argmax(1) == ml_preds.argmax(1)
            else:
                lnr_mask = self.lnr_mask[idxs]
            
            inputs = inputs[lnr_mask]
            targets = targets[lnr_mask]
            ml_preds = ml_preds[lnr_mask]

            outputs = self.net(inputs)
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


class SelfMLSTDTrainer(Trainer):
    """For faster compute the ml predictions are stored. Thus, assumes there is no data augmentation"""
    def __init__(self, args, net, trainloader, testloader, optimizer, scheduler, sch_updt, slow_rate=5):
        super().__init__(args, net, trainloader, testloader, optimizer, scheduler, sch_updt, slow_rate)
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
            outputs = self.net(inputs)
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