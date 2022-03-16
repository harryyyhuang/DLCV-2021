import os
import argparse

import torch
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
from sklearn.preprocessing import LabelEncoder

import csv
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

from dataloader import OfficeDataset
from model import DownStreamModel, AG, REST

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(SEED)
np.random.seed(SEED)

# check device
def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else 'cpu'

def parse_args():
    parser = argparse.ArgumentParser(description="Few shot learning")
    parser.add_argument('--load', type=str, help="Model checkpoint path")
    parser.add_argument('--train_csv', type=str, help="Training images csv file")
    parser.add_argument('--train_data_dir', type=str, help="Training images directory")
    parser.add_argument('--epochs', type=int, default=1, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size for training")
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

    le = LabelEncoder()
    le.classes_ = np.load("classes.npy", allow_pickle=True)
    val_dataset = OfficeDataset(args.test_csv, args.test_data_dir, le)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=False)

    resnet = models.resnet50(pretrained=False).to(device)
    resnet.fc = REST()
    # resnet.load_state_dict(torch.load("./resnet.pth")["resnet"]) 
    resnet.load_state_dict(torch.load("restnetC_new.pth"))
    

    down_stream = DownStreamModel().to(device)
    down_stream.load_state_dict(torch.load("./classifireC_new.pth")) 



    # for epoch in range(args.epochs):


    valid_loss = []
    valid_accs = []
    resnet.eval()
    down_stream.eval()
    out_list = []
    name_list = []
    for (img, label, name) in tqdm(val_loader):

        img = img.to(device)

        img = resnet(img)
        predict = down_stream(img)


        # opt.zero_grad()
        # loss.backward()
        # opt.step()

        out = predict.argmax(dim=1)
        out_list.append(out)
        name_list.append(name)
        acc = (out == label.to(device)).float().mean()

        valid_accs.append(acc)

    valid_accs = sum(valid_accs) / len(valid_accs)

    # print(out_list)
    print(f"[ Valid |/{args.epochs:03d} ], acc = {valid_accs:.5f}")

    with open(args.output_csv, 'w') as csvfile:
        csv_writer = csv.writer(csvfile)
        field = ['id', 'filename', 'label']
        csv_writer.writerow(field)

        for i, data in enumerate(zip(out_list, name_list)):
            
            label = le.inverse_transform(data[0].tolist())
            name = data[1]
            # print(le.inverse_transform(label))
            # print(name)

            # data = [str(x) for x in data]
            # data.insert(0,  str(i))
            csv_writer.writerow([i, str(name[0]), label[0]])

    
    # torch.save(resnet.state_dict(), './improved-net.pt')

