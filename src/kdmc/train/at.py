from numpy import dtype
import torch
from tqdm import tqdm
import wandb
from kdmc.attack.core import parse_attack
from kdmc.data.core import get_num_classes
from kdmc.train.base import Trainer
import torch.nn.functional as F


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
            adv_inputs = self.atk(inputs, targets, snr)
            outputs = self.net(adv_inputs)

            ml_preds = self.get_adv_ml_preds(batch, adv_inputs)
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
