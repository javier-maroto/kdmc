import torch
from tqdm import tqdm
from kdmc.attack.core import parse_attack
from kdmc.train.base import KTTrainer
import torch.nn.functional as F
import wandb
import torch.nn as nn

class RSLADTrainer(KTTrainer):

    def __init__(self, args, net, trainloader, testloader, optimizer, scheduler, sch_updt, slow_rate=5):
        super().__init__(args, net, trainloader, testloader, optimizer, scheduler, sch_updt, slow_rate)
        self.atk = parse_attack(self.net, args.atk)
        self.clamp = True
        self.lmbd = args.rslad_lmbd  # Between 0 and 1

    def train(self, epoch):
        print('\nEpoch: %d' % epoch)
        self.net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(tqdm(self.train_dl)):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            kt_logits = self.logits_kt(inputs)  
            kt_preds = torch.sum(torch.stack(
                [F.softmax(l, dim=-1) * b for l, b in zip(kt_logits, self.beta)]), dim=0)
            kt_targets = (
                self.alpha * kt_preds 
                + (1 - self.alpha) * F.one_hot(targets, num_classes=kt_preds.shape[-1])).detach()

            self.optimizer.zero_grad()
            # We only accept first model logits just in case
            adv_logits = self.rslad_inner_loss(
                self.net, kt_logits[0], inputs, targets, self.optimizer, step_size=self.atk.alpha,
                epsilon=self.atk.eps, perturb_steps=self.atk.steps)
            nat_logits = self.net(inputs)

            kl_Loss1 = self.kl_loss(F.log_softmax(adv_logits,dim=1), kt_targets)
            kl_Loss2 = self.kl_loss(F.log_softmax(nat_logits,dim=1), kt_targets)
            kl_Loss1 = torch.mean(kl_Loss1)
            kl_Loss2 = torch.mean(kl_Loss2)

            loss = (1 - self.lmbd) * kl_Loss1 + self.lmbd * kl_Loss2
            loss.backward()
            self.optimizer.step()
            if self.sch_updt == 'step':
                self.scheduler.step()

            train_loss += loss.item()
            _, predicted = adv_logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        wandb.log({'train.acc': 100.*correct/total, 'train.loss': train_loss/(batch_idx+1), 'epoch': epoch})

    def kl_loss(self, a, b):
        loss = -a*b + torch.log(b+1e-5)*b
        return loss
    
    def rslad_inner_loss(self, model,
                    teacher_logits,
                    x_natural,
                    y,
                    optimizer,
                    step_size=0.003,
                    epsilon=8./255,
                    perturb_steps=10,
                    beta=6.0):
        # define KL-loss
        criterion_kl = nn.KLDivLoss(size_average=False,reduce=False)
        model.eval()
        batch_size = len(x_natural)
        # generate adversarial example
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()

        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                        F.softmax(teacher_logits, dim=1))
                loss_kl = torch.sum(loss_kl)
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            if self.clamp:
                x_adv = torch.clamp(x_adv, 0.0, 1.0)

        model.train()

        x_adv = torch.autograd.Variable(x_adv, requires_grad=False)
        # zero gradient
        optimizer.zero_grad()
        logits = model(x_adv)
        return logits