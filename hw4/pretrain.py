import os
import argparse

import torch
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler

import csv
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

from dataloader import MiniTrainDataset, PrototypicalBatchSampler

from byol_pytorch import BYOL
from model import REST


# check device
def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else 'cpu'

def parse_args():
    parser = argparse.ArgumentParser(description="Few shot learning")
    parser.add_argument('--load', type=str, help="Model checkpoint path")
    parser.add_argument('--train_csv', type=str, help="Training images csv file")
    parser.add_argument('--train_data_dir', type=str, help="Training images directory")
    parser.add_argument('--epochs', type=int, default=200, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for training")
    parser.add_argument('--test_csv', type=str, help="Testing images csv file")
    parser.add_argument('--test_data_dir', type=str, help="Testing images directory")
    parser.add_argument('--testcase_csv', type=str, help="Test case csv")
    parser.add_argument('--output_csv', type=str, help="Output filename")

    return parser.parse_args()

if __name__=='__main__':
    args = parse_args()

    # get device
    device = get_device()
    print(f'DEVICE: {device}')

    train_dataset = MiniTrainDataset(args.train_csv, args.train_data_dir)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=False)

    resnet = models.resnet50(pretrained=False).to(device)
    resnet.fc = REST()
    # resnet.load_state_dict(torch.load("./improved-net_no_pretrain.pt"))

    learner = BYOL(
        resnet,
        image_size = 128,
        hidden_layer = 'avgpool'
    ).to(device)

    opt = torch.optim.Adam(learner.parameters(), lr=0.001)

    old_loss = 1000

    for epoch in range(args.epochs):
        
        train_loss = []

        for (img, label) in tqdm(train_loader):

            img = img.to(device)

            loss = learner(img)

            opt.zero_grad()

            loss.backward()

            opt.step()

            learner.update_moving_average()

            train_loss.append(loss.item())
        avg_loss = np.mean(train_loss)
        if(avg_loss < old_loss):
            old_loss = avg_loss
        print('Avg Train Loss: {}, Old Train Loss: {}'.format(avg_loss, old_loss))

    torch.save(resnet.state_dict(), './improved-net_no_pretrain.pt')

