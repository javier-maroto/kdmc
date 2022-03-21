from tqdm import tqdm
import wandb
from kdmc.attack.core import parse_attack
from kdmc.train.base import KTTrainer
from kdmc.utils import softXEnt
import torch.nn.functional as F


class AKDTrainer(KTTrainer):

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
            adv_inputs = self.atk(inputs, targets)
            outputs = self.net(adv_inputs)
            self.optimizer.zero_grad()
            kt_preds = self.pred_kt(adv_inputs)
            kt_targets = self.alpha * kt_preds + (1 - self.alpha) * F.one_hot(targets, num_classes=kt_preds.shape[-1])
            loss = softXEnt(outputs, kt_targets)
            loss.backward()
            self.optimizer.step()
            if self.sch_updt == 'step':
                self.scheduler.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        wandb.log({'train.acc': 100.*correct/total, 'train.loss': train_loss/(batch_idx+1), 'epoch': epoch})
