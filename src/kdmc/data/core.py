from torch.utils.data import DataLoader

from kdmc.data.rml2016_10a import get_rml2016_10a_datasets


def get_datasets(args):
    if args.dataset == 'rml2016.10a':
        return get_rml2016_10a_datasets(f'{args.root_path}/data')
    else:
        raise NotImplementedError(f"dataset not implemented: {args.dataset}")


def get_num_classes(dataset):
    if dataset == 'rml2016.10a':
        return 11
    else:
        raise NotImplementedError(f"dataset not implemented: {args.dataset}")


def get_input_dims(args):
    pass


def create_dataloaders(args, trainset, testset, num_workers=0):
    trainloader = DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
    testloader = DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)
    return trainloader, testloader