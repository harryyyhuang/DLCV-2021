import numpy as np
from torch.utils.data import DataLoader
from data_loader import build_data
from model import build_model
import torch
import torch.nn as nn
import argparse
import math
from torch.utils.data import ConcatDataset, DataLoader, Subset, random_split
from torch.autograd import Variable
import os
import torchvision.models as models
import torchvision
from PIL import Image
from torch.autograd import grad as torch_grad
import matplotlib.pyplot as plt
import torch.nn.functional as F
import csv

# check device
def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else 'cpu'

# fix random seed
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def write_p1(args, name_list, label_list):
    with open(args.out_path, 'w') as csvfile:
        csv_writer = csv.writer(csvfile)
        field = ['image_id', 'label']
        csv_writer.writerow(field)

        for (name, label) in zip(name_list, label_list):
            csv_writer.writerow([name[0], label.cpu().numpy()[0]])

# fix random seed for reproducibility
same_seeds(116)
parser = argparse.ArgumentParser(
description="training parameter")
parser.add_argument("--dataset", default="p3_test")
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--data_path", default="./hw2_data/digits/mnistm/test", type=str)
parser.add_argument("--data_name", default="mnistm", type=str)
parser.add_argument("--out_path", default="./mnist.cvs", type=str)
parser.add_argument("--model", default="p3_test")
parser.add_argument("--epoch", default=1, type=int)
parser.add_argument("--model_path", default="./model_pth", type=str)

parser.add_argument("--workspace_dir", default=".", type=str)
args = parser.parse_args()

# get device
device = get_device()
print(f'DEVICE: {device}')


""" Medium: WGAN, 50 epoch, n_critic=5, clip_value=0.01 """
n_epoch = args.epoch # 50

# Model
model_1 = build_model(args)
model_1.to(device)
model_1.eval()


# DataLoader
test_dataset = build_data(args, "whatever")
dataloader_1 = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)


steps = 0
domain_acc = 0
for e, epoch in enumerate(range(n_epoch)):
    # progress_bar = qqdm(dataloader_train)
    name_list = []
    label_list = []
    for data in (dataloader_1):
        imgs, name = data
        imgs = imgs.cuda()
        
        cls_logit, domain_logit = model_1(imgs)
        domain_class_out = cls_logit.argmax(dim=1)

        label_list.append(domain_class_out)
        name_list.append(name)


    
    # name_list = torch.cat(name_list, 0)


    write_p1(args, name_list, label_list)
        

    