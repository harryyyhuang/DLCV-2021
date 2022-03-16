# from comet_ml import Experiment
from torchvision.transforms.transforms import Normalize

# # Create an experiment with your api key
# experiment = Experiment(
#     api_key="gbIbPvPkSfvA1rSVHFO2B1qAD",
#     project_name="general",
#     workspace="harryyyhuang",
# )

import numpy as np
from torch.utils.data import DataLoader
from data_loader import build_data
from model import AutoED, build_model
import torch
import torch.nn as nn
import argparse
import math
from torch.utils.data import ConcatDataset, DataLoader, Subset, random_split
from torch.autograd import Variable
import os
from model import Generator, Discriminator
import torchvision.models as models
import torchvision
from PIL import Image
from qqdm import qqdm
from torch.autograd import grad as torch_grad
import matplotlib.pyplot as plt
import torch.nn.functional as F

# check device
def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser(
description="training parameter")
parser.add_argument("--p1_path", default="./hw2_data/face", type=str)
parser.add_argument("--p1_test_path", default="./hw2_data/face/test", type=str)
parser.add_argument("--p1_gen_path", default="./gen", type=str)
parser.add_argument("--p2_path", default="./hw2_data/digits", type=str)
parser.add_argument("--dataset", default="p1")
parser.add_argument("--model", default="AutoEncoder")
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--epoch", default=30, type=int)
parser.add_argument("--model_path", default="./model_pth", type=str)
parser.add_argument("--p2_out_path", default="./train/predict", type=str)
parser.add_argument("--workspace_dir", default=".", type=str)
args = parser.parse_args()


# fix random seed
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# fix random seed for reproducibility
same_seeds(52)
# get device
device = get_device()
print(f'DEVICE: {device}')

# You may replace the workspace directory if you want.
workspace_dir = args.workspace_dir

# Training hyperparameters
# batch_size = 64

lr = 0.001

""" Medium: WGAN, 50 epoch, n_critic=5, clip_value=0.01 """
n_epoch = args.epoch # 50


# Model
AutoEnDe = build_model(args)


# Loss
criterion = nn.BCELoss()
criterion.size_average = False


""" Medium: Use RMSprop for WGAN. """
# Optimizer
opt_AutoEnDe = torch.optim.Adam(AutoEnDe.parameters(), lr=lr)
# opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
# opt_AutoEnDe = torch.optim.RMSprop(AutoEnDe.parameters(), lr=lr)
# opt_G = torch.optim.RMSprop(G.parameters(), lr=lr)

# DataLoader
train_dataset = build_data(args, "train")
dataloader_train = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
test_dataset = build_data(args, "test")
dataloader_test = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
steps = 0
step_val = 0
for e, epoch in enumerate(range(n_epoch)):
    progress_bar = qqdm(dataloader_train)
    AutoEnDe.train()
    train_loss = []
    for i, data in enumerate(progress_bar):
        imgs = data
        imgs = imgs.cuda()

        bs = imgs.size(0)

        opt_AutoEnDe.zero_grad()
        reconstruct, mean_var = AutoEnDe(data.to(device))
        mu = mean_var[:, :128, :, :]
        var = mean_var[:, 128:, :, :]
        # print(reconstruct)
        # print(data.shape)
        # exit()

        
        kl_div = - 0.5 * torch.sum(1 + var  - mu.pow(2) - var.exp())
        
        loss = criterion(reconstruct, data.to(device)) + kl_div
        loss.backward()
        opt_AutoEnDe.step()

        steps += 1
        # print(steps)
        # Set the info of the progress bar
        #   Note that the value of the GAN loss is not directly related to
        #   the quality of the generated images.
        progress_bar.set_infos({
            'Loss_AUTO': round(loss.item(), 4),
            # 'Loss_G': round(loss_G.item(), 4),
            'Epoch': e+1,
            'Step': steps,
        })
        train_loss.append(loss.item())
    train_loss = sum(train_loss) / len(train_loss)
    # experiment.log_metric("train_loss", train_loss, step=epoch)



    progress_bar = qqdm(dataloader_test)
    AutoEnDe.eval()
    test_loss = []
    for i, data in enumerate(dataloader_test):

        reconstruct, mean_var = AutoEnDe(data.to(device))
        mu = mean_var[:, :128, :, :]
        var = mean_var[:, 128:, :, :]
        # print(reconstruct)
        # print(data.shape)
        # exit()

        # tf.nn.sigmoid_cross_entropy_with_logits
        
        kl_div = - 0.5 * torch.sum(1 + var  - mu.pow(2) - var.exp())
        
        loss = criterion(reconstruct, data.to(device)) + kl_div

        step_val += 1
        # print(step_val)
        # Set the info of the progress bar
        #   Note that the value of the GAN loss is not directly related to
        #   the quality of the generated images.
        progress_bar.set_infos({
            'Loss_AUTO': round(loss.item(), 4),
            # 'Loss_G': round(loss_G.item(), 4),
            'Epoch': e+1,
            'Step': step_val,
        })
        test_loss.append(loss.item())

        if(i == 0):
            # f_imgs_sample = (logits.data + 1) / 2.0
            reconstruct = (reconstruct.data + 1) / 2.0

            for k in range(args.batch_size):
                filename = os.path.join(args.p1_gen_path, f'{k}_auto.png')
                torchvision.utils.save_image(reconstruct[k], filename) 
                print(f' | Save samples to {filename}.')
    test_loss = sum(test_loss) / len(test_loss)
    # experiment.log_metric("test_loss", test_loss, step=epoch)

torch.save(AutoEnDe, "./model/AE.pth")

