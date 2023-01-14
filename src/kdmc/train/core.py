import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import ExponentialLR, OneCycleLR, ConstantLR
from kdmc.data.core import get_num_classes
from kdmc.model.cldnn import CLDNN
from kdmc.model.cnn import CNN
from kdmc.train.akd import AKDTrainer
from kdmc.train.at import AMLATTrainer, ATTrainer, GLNRATTrainer, GMLATTrainer, LNRATTrainer, MLATTrainer, MMLATTrainer, SelfMLATTrainer, YAMLATTrainer, YGMLATTrainer, YMLATTrainer
from kdmc.train.base import MLTrainer
from kdmc.train.rslad import RSLADTrainer
from kdmc.train.std import LNRSTDTrainer, MLSTDTrainer, MMLSTDTrainer, STDTrainer, SelfMLSTDTrainer, YMLSTDTrainer
from kdmc.model.resnet import ResNet_OShea


def create_model(args, num_classes=None):
    if num_classes is None:
        num_classes = get_num_classes(args.dataset)
    if args.arch == 'resnet':
        net = ResNet_OShea(num_classes, args.time_samples)
    elif args.arch == 'cldnn':
        net = CLDNN(args.time_samples, num_classes)
    elif args.arch == 'cnn':
        net = CNN(args.time_samples, num_classes)
    else:
        raise NotImplementedError(f"arch not implemented: {args.arch}")
    net = net.to(args.device)
    if args.device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    return net


def create_scheduler(optimizer, args, max_epochs, steps_epoch):
    if args.sched == 'exp':
        scheduler = ExponentialLR(optimizer, gamma=args.sch_gamma)
        sch_update = 'epoch'
    elif args.sched == '1cyl':
        scheduler = OneCycleLR(optimizer, max_lr=0.21, epochs=max_epochs, steps_per_epoch=steps_epoch)  # has to be updated every batch
        sch_update = 'step'
    elif args.sched == 'fixed':
        scheduler = ConstantLR(optimizer, factor=1, last_epoch=-1)
        sch_update = 'epoch'
    else:
        raise NotImplementedError(f"scheduler not implemented: {args.sched}")
    return scheduler, sch_update


def resume_from_checkpoint(path, net, optimizer, scheduler):
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(f'./checkpoint/{path}/ckpt_last.pth')
    net.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optim'])
    scheduler.load_state_dict(checkpoint['sched'])
    start_epoch = checkpoint['epoch']
    return start_epoch, net, optimizer, scheduler


def get_trainer(args, net, trainloader, testloader, optimizer, scheduler, sch_updt, slow_rate=5):
    if args.loss == 'std':
        return STDTrainer(args, net, trainloader, testloader, optimizer, scheduler, sch_updt, slow_rate=slow_rate)
    elif args.loss == 'std_ml':
        if args.dataset in ('sm_rml2018'):
            return MMLSTDTrainer(args, net, trainloader, testloader, optimizer, scheduler, sch_updt, slow_rate=slow_rate)
        return MLSTDTrainer(args, net, trainloader, testloader, optimizer, scheduler, sch_updt, slow_rate=slow_rate)
    elif args.loss == 'std_yml':
        return YMLSTDTrainer(args, net, trainloader, testloader, optimizer, scheduler, sch_updt, slow_rate=slow_rate)
    elif args.loss == 'std_lnr':
        return LNRSTDTrainer(args, net, trainloader, testloader, optimizer, scheduler, sch_updt, slow_rate=slow_rate)
    elif args.loss == 'at':
        return ATTrainer(args, net, trainloader, testloader, optimizer, scheduler, sch_updt, slow_rate)
    elif args.loss == 'at_ml':
        if args.dataset in ('sm_rml2018'):
            return MMLATTrainer(args, net, trainloader, testloader, optimizer, scheduler, sch_updt, slow_rate=slow_rate)
        if args.dataset in ('sbasic', 'sawgn', 'sawgn2p'):
            return GMLATTrainer(args, net, trainloader, testloader, optimizer, scheduler, sch_updt, slow_rate=slow_rate)
        else:
            return MLATTrainer(args, net, trainloader, testloader, optimizer, scheduler, sch_updt, slow_rate=slow_rate)
    elif args.loss == 'at_aml':
        return AMLATTrainer(args, net, trainloader, testloader, optimizer, scheduler, sch_updt, slow_rate=slow_rate)
    elif args.loss == 'at_yml':
        if args.dataset in ('sbasic', 'sawgn', 'sawgn2p'):
            return YGMLATTrainer(args, net, trainloader, testloader, optimizer, scheduler, sch_updt, slow_rate=slow_rate)
        else:
            return YMLATTrainer(args, net, trainloader, testloader, optimizer, scheduler, sch_updt, slow_rate=slow_rate)
    elif args.loss == 'at_yaml':
        return YAMLATTrainer(args, net, trainloader, testloader, optimizer, scheduler, sch_updt, slow_rate=slow_rate)
    elif args.loss == 'at_lnr':
        if args.dataset in ('sbasic', 'sawgn', 'sawgn2p'):
            return GLNRATTrainer(args, net, trainloader, testloader, optimizer, scheduler, sch_updt, slow_rate=slow_rate)
        else:
            return LNRATTrainer(args, net, trainloader, testloader, optimizer, scheduler, sch_updt, slow_rate=slow_rate)
    elif args.loss == 'at_sml':
        return SelfMLATTrainer(args, net, trainloader, testloader, optimizer, scheduler, sch_updt, slow_rate=slow_rate)
    elif args.loss == 'std_sml':
        return SelfMLSTDTrainer(args, net, trainloader, testloader, optimizer, scheduler, sch_updt, slow_rate=slow_rate)
    elif args.loss == 'ml':
        return MLTrainer(args, net, trainloader, testloader, optimizer, scheduler, sch_updt, slow_rate=slow_rate)
    elif args.loss == 'akd':
        return AKDTrainer(args, net, trainloader, testloader, optimizer, scheduler, sch_updt, slow_rate)
    elif args.loss == 'rslad':
        return RSLADTrainer(args, net, trainloader, testloader, optimizer, scheduler, sch_updt, slow_rate)
    else:
        raise NotImplementedError(f"loss not implemented: {args.loss}")
