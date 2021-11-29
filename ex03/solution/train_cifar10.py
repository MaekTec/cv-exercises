import torch
import argparse
import torchvision
from custom_datasets import CIFAR10_custom as CIFAR10
from networks.solution import LeNet_solution
from networks.LeNet import LeNet
from util import AverageMeter, accuracy
from torch.utils.tensorboard import SummaryWriter
from augmentations import horizontal_flip, random_resize_crop
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--lr', type=int, default=0.03, help='learning rate')
    parser.add_argument('--num_epochs', type=int, default=256, help='number of epochs to train')
    parser.add_argument('--use_solution_network', type=bool, default=False, help='use the network you implemented?')
    parser.add_argument('--subsample_factor', type=float, default=0.2,
                        help='decreases dataset size to speed up overfitting. Keep fixed for exercise. You can play with this in the end')
    parser.add_argument('--datafolder', type=str, default='./data',
                        help='path to cifar')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='path to cifar')
    parser.add_argument('--transforms', type=str, default='basic',
                        help='which transformations to use', choices=['basic', 'own', 'torchvision'])
    parser.add_argument('--run_name', type=str, default='run1',
                        help='for tensorboard logging')
    parser.add_argument('--scale', type=float, default=[1.0, 1.6], nargs='+',
                        help='scale for resize')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of data loader threads per dataloader. 0 will use the main thread and is good for debugging')
    args = parser.parse_args()
    return args

def get_transofrms():
    if args.transforms == 'basic':
        train_transforms = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    elif args.transforms == 'own':
        # todo: should be implemented by studens
        train_transforms = torchvision.transforms.Compose(
            [horizontal_flip(p=0.5),
             random_resize_crop(size=32, scale=args.scale),
             torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    elif args.transforms == 'torchvision':
        # todo: should be implemented by studens
        raise NotImplementedError

    val_transforms = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    return train_transforms, val_transforms

def get_dataloaders(args, train_transforms, val_transforms):

    # define dataset
    trainset = CIFAR10(root=args.datafolder, subsample_factor=args.subsample_factor,
                       download=True, transform=train_transforms,
                       split='train')
    valset = CIFAR10(root=args.datafolder, subsample_factor=args.subsample_factor,
                       download=True, transform=val_transforms,
                       split='val')
    testset = CIFAR10(root=args.datafolder, subsample_factor=args.subsample_factor,
                       download=True, transform=val_transforms,
                       split='test')
    #get dataloader
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=0)

    # get dataloader
    val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                                              shuffle=False, num_workers=0)
    #get test_loader
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                              shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader

def train_one_epoch(model, train_loader, loss_fn, optimizer, epoch, args):
    model.train()

    #keep track of interesting values
    losses = AverageMeter()
    top1 = AverageMeter()

    for idx, (images, labels) in enumerate(train_loader):
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        output = model(images)

        loss = loss_fn(output, labels)
        losses.update(loss)
        acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        top1.update(acc1[0], args.batch_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #this is todo for students
        if (idx + 1) % 100 == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), loss=losses, top1=top1))

    return losses.avg, top1.avg

def validate(model, val_loader, loss_fn, epoch, args, setting='validate'):
    #todo by sutdents
    # turn off updates of non-trainable parameters. i.e. batch-norm
    model.eval()
    # this is todo for students
    #keep track of interesting values
    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        for idx, (images, labels) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            output = model(images)

            loss = loss_fn(output, labels)
            losses.update(loss)
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            top1.update(acc1[0], args.batch_size)


        print(setting + ': [{0}][{1}/{2}]\t'
              '({loss.avg:.3f})\t'
              '({top1.avg:.3f})'.format(
               epoch, idx + 1, len(val_loader), loss=losses, top1=top1))

    return losses.avg, top1.avg


def set_optimizer(model, args):
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.lr)

    return optimizer

if __name__ == '__main__':
    args = parse_args()
    #define augmentation and get data loaders
    train_transorms, val_transforms = get_transofrms()
    train_loader, val_loader, test_loader = get_dataloaders(args, train_transorms, val_transforms)

    tb_writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.run_name)) #todo by students

    #get model
    if args.use_solution_network:
        model = LeNet_solution()
    else:
        model = LeNet()
    model.cuda()

    #define optimizer
    optimizer = set_optimizer(model, args)

    #define loss
    loss_fn = torch.nn.CrossEntropyLoss().cuda()

    for epoch in range(5):
        train_loss, train_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, epoch, args )

        # todo by students
        # tensorboard logger
        tb_writer.add_scalar('train_loss', train_loss, epoch)
        tb_writer.add_scalar('train_acc', train_acc, epoch)

        val_loss, val_acc = validate(model, val_loader, loss_fn, epoch, args, 'validate')

        # todo by students
        # tensorboard logger
        tb_writer.add_scalar('val_loss', val_loss, epoch)
        tb_writer.add_scalar('val_acc', val_acc, epoch)

    test_loss, test_acc = validate(model, test_loader, loss_fn, epoch, args, 'test')




