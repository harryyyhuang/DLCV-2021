import os
import argparse

import torch
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler

import csv
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

from dataloader import OfficeDataset
from model import DownStreamModel, AG, REST


# check device
def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else 'cpu'

def parse_args():
    parser = argparse.ArgumentParser(description="Few shot learning")
    parser.add_argument('--load', type=str, help="Model checkpoint path")
    parser.add_argument('--train_csv', type=str, help="Training images csv file")
    parser.add_argument('--train_data_dir', type=str, help="Training images directory")
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training")
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

    train_dataset = OfficeDataset(args.train_csv, args.train_data_dir)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=False)

    val_dataset = OfficeDataset(args.test_csv, args.test_data_dir, train_dataset.le)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=False)

    resnet = models.resnet50(pretrained=False).to(device)
    resnet.fc = REST()
    resnet.load_state_dict(torch.load("./resnet.pth")["resnet"]) 
    # resnet.load_state_dict(torch.load("restnetC_no.pth"))
    

    down_stream = DownStreamModel().to(device)
    # down_stream.load_state_dict(torch.load("classifireC_no.pth"))
    # augmentation = AG().to(device)

    opt = torch.optim.Adam(params=[
            {'params': resnet.parameters()},
            {'params': down_stream.parameters()}
        ], lr=0.0001)

    criterion = nn.CrossEntropyLoss()

    old_val_acc = 0

    for epoch in range(args.epochs):

        train_loss = []
        train_accs = []
        resnet.eval()
        down_stream.train()
        
        for (img, label) in tqdm(train_loader):

            img = img.to(device)
            # img = augmentation(img)
            # print(img.shape)
            img = resnet(img)
            predict = down_stream(img)

            loss = criterion(predict, label.to(device))

            opt.zero_grad()
            loss.backward()
            opt.step()

            out = predict.argmax(dim=1)
            acc = (out == label.to(device)).float().mean()

            train_loss.append(loss.item())
            train_accs.append(acc)

        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_accs) / len(train_accs)

        print(f"[ Train | {epoch + 1:03d}/{args.epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

        valid_loss = []
        valid_accs = []
        resnet.eval()
        down_stream.eval()

        for (img, label) in tqdm(val_loader):

            img = img.to(device)

            img = resnet(img)
            predict = down_stream(img)

            loss = criterion(predict, label.to(device))

            # opt.zero_grad()
            # loss.backward()
            # opt.step()

            out = predict.argmax(dim=1)
            acc = (out == label.to(device)).float().mean()

            valid_loss.append(loss.item())
            valid_accs.append(acc)

        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_accs = sum(valid_accs) / len(valid_accs)


        print(f"[ Valid | {epoch + 1:03d}/{args.epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_accs:.5f}")
        if(valid_accs > old_val_acc):
            print("saveing the model for prob 1...")
            # torch.save(resnet.state_dict(), "restnetC_new.pth")
            # torch.save(down_stream.state_dict(), "classifireC_new.pth")
            old_val_acc = valid_accs

    
    # torch.save(resnet.state_dict(), './improved-net.pt')

