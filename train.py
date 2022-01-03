import sys
import getopt
from torch.utils.data import Dataset, DataLoader
from byol import BYOL
from torchvision import models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch
import random
import pandas as pd
import os
from tqdm import tqdm
import PIL


datapath = '..'
argvs = sys.argv[1:]
try:
    opts, args = getopt.getopt(argvs, "", ["data="])
    for opt, arg in opts:
        if opt == '--data':
            datapath = arg
except getopt.GetoptError:
    print('train.py --data <data_path> ')
    sys.exit(2)
print(f'datapath[{datapath}]')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def train(train_loader, model, optimizer, scheduler, epoch):
    losses = AverageMeter('Loss', ':.4f')

    model.train()

    for images, _ in tqdm(train_loader):

        loss = model([images[0].to(device), images[1].to(device)])
        losses.update(loss.item(), images[0].size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        model.update_moving_average()  # update moving average of target encoder

    torch.save(model.state_dict(), f'checkpoint_{epoch}.pth')
    print(losses)


class TwoCropsTransform:
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class GaussianBlur(object):
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(PIL.ImageFilter.GaussianBlur(radius=sigma))
        return x


class TrainDataSet(Dataset):
    def __init__(self, images_folder_path,  transform=None):
        self.images_folder_path = images_folder_path
        self.images_list = os.listdir(images_folder_path)
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, index):
        image_name = self.images_list[index]
        image = PIL.Image.open(os.path.join(
            self.images_folder_path, image_name))
        return self.transform(image), 0


train_augmentation = [
    transforms.RandomResizedCrop(256, scale=(0.5, 1.0)),
    transforms.RandomRotation(45),
    transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
    ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.62044597, 0.45018077, 0.6520328],
                         std=[0.13870312, 0.17680055, 0.16805622])
]

divide_transform = TwoCropsTransform(transforms.Compose(train_augmentation))

batch_size = 4

train_dataset = TrainDataSet(
    datapath+'/train', transform=divide_transform)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)

resnet = models.resnet50(pretrained=True)

model = BYOL(
    resnet,
    image_size=256,
    hidden_layer='avgpool'
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)

model.to(device)

epoches = 30

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, eta_min=1e-5, T_max=30)

for epoch in range(epoches):
    print(f'epoce: {epoch}')
    train(train_loader, model, optimizer, scheduler, epoch)

torch.save(model.state_dict(), datapath+'/state.pt')
