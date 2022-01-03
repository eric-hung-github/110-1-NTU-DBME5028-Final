import sys
import getopt
import PIL
from tqdm import tqdm
import os
import pandas as pd

import torch
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import Dataset, DataLoader

from byol import BYOL

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


class TestDataSet(Dataset):
    def __init__(self, images_folder_path, pair_dic, transform=None):
        self.images_folder_path = images_folder_path
        self.pair_dic = pair_dic
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.pair_dic)

    def __getitem__(self, index):
        element = self.pair_dic.iloc[index]
        img_name = (element[0], element[1])
        label = str(element[0].split('.')[0]+'_'+element[1].split('.')[0])

        images = []
        for i in range(2):
            images.append(PIL.Image.open(os.path.join(
                self.images_folder_path, img_name[i])))
            images[i] = self.transform(images[i])

        return images[0], images[1], label


def test(test_loader, model, threshold):

    predicts = []

    model.eval()
    with torch.no_grad():
        for images1, images2, label in tqdm(test_loader):
            loss = model(
                [images1.to(device), images2.to(device)], predict=True)
            predicts.append((label[0], 1 if loss < threshold else 0))
    return predicts


test_augmentation = [
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.6481022, 0.49916938, 0.70643044],
                         std=[0.13052933, 0.16704792, 0.15464625])
]

test_transform = transforms.Compose(test_augmentation)

resnet = models.resnet50(pretrained=True)

model = BYOL(
    resnet,
    image_size=256,
    hidden_layer='avgpool'
)

model.load_state_dict(torch.load(
    datapath+'/model_state.pth.tar')['state_dict'])

model.to(device)

test_csv = pd.read_csv(datapath+'/queries.csv', names=['img1', 'img2'])
test_set = TestDataSet(datapath+'/test', test_csv,
                       transform=test_transform)
test_loader = DataLoader(test_set)

test_result = test(test_loader, model, 1.1)
df = pd.DataFrame(data=test_result, columns=['query', 'prediction'])
df.to_csv(datapath+'/submit.csv', index=False)
