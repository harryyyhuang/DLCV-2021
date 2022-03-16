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
from model import Generator, Discriminator
from inceptionscorepytorch.inception_score import inception_score
import torchvision.models as models
import torchvision
from PIL import Image
from qqdm import qqdm
from pytorch_fid.fid_score import calculate_fid_given_paths
from torch.autograd import grad as torch_grad
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
from sklearn import manifold, datasets
import matplotlib.colors as mcolors

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

# fix random seed for reproducibility
same_seeds(1029)
parser = argparse.ArgumentParser(
description="training parameter")
parser.add_argument("--p1_path", default="./hw2_data/face", type=str)
parser.add_argument("--p1_test_path", default="./hw2_data/face/test", type=str)
parser.add_argument("--p1_gen_path", default="./gen", type=str)
parser.add_argument("--p2_gen_path", default="./gen_p2", type=str)
parser.add_argument("--p2_path", default="./hw2_data/digits/", type=str)
parser.add_argument("--dataset", default="p3")
parser.add_argument("--batch_size", default=512, type=int)
parser.add_argument("--model", default="p3")
parser.add_argument("--epoch", default=100, type=int)
parser.add_argument("--model_path", default="./model_pth", type=str)
parser.add_argument("--p2_out_path", default="./train/predict", type=str)
parser.add_argument("--workspace_dir", default=".", type=str)
args = parser.parse_args()

# get device
device = get_device()
print(f'DEVICE: {device}')

# You may replace the workspace directory if you want.
workspace_dir = args.workspace_dir

# Training hyperparameters
# batch_size = 64
z_dim = 100
lr = 0.0001 

""" Medium: WGAN, 50 epoch, n_critic=5, clip_value=0.01 """
n_epoch = args.epoch # 50
n_critic = 2 # 5
clip_value = 0.01
lambda_norm = 10.0
test_len = 100
log_dir = os.path.join(workspace_dir, 'logs')
ckpt_dir = os.path.join(workspace_dir, 'checkpoints')
os.makedirs(log_dir, exist_ok=True)
os.makedirs(ckpt_dir, exist_ok=True)
model = build_model(args)
model.to(device)
model.eval()


# opt_D = torch.optim.RMSprop(D.parameters(), lr=lr)
# opt_G = torch.optim.RMSprop(G.parameters(), lr=lr)

# DataLoader
train_dataset, test_dataset = build_data(args, "usps2svhn")
mnist_test = build_data(args, "usps")
usps_test = build_data(args, "mnist")
dataloader_mnist = DataLoader(mnist_test, batch_size=32, shuffle=True, num_workers=2)
dataloader_usps = DataLoader(usps_test, batch_size=32, shuffle=True, num_workers=2)
# dataloader_treu_test = DataLoader(true_test, batch_size=args.batch_size, shuffle=True, num_workers=2)

steps = 0
domain_acc = 0

plt.figure(figsize=(8, 8))
tsne_array = []
label_array = []
for data in dataloader_usps:
    imgs = data[0].cuda()
    label = data[1]

    features = model(imgs)
    label = torch.zeros((label.shape[0]))
    tsne_array.append(features.cpu().detach())
    label_array.append(label.cpu().detach())

tsne_data = torch.cat(tsne_array, 0).numpy()
label_data = torch.cat(label_array, 0).numpy()
print(label_data.shape)
x_tsne = manifold.TSNE(n_components=2, init='random', random_state=5, verbose=1).fit_transform(tsne_data)
x_min, x_max = x_tsne.min(0), x_tsne.max(0)

X_norm = (x_tsne - x_min) / (x_max - x_min) 
# print(X_norm.shape)
# print(label.shape)


for i in range(X_norm.shape[0]):
        plt.text(X_norm[i, 0], X_norm[i, 1], str(int(label_data[i])), color=plt.cm.Set3(int(label_data[i])),
            fontdict={'weight': 'bold', 'size': 9})


tsne_array = []
label_array = []
for data in dataloader_mnist:
    imgs = data[0].cuda()
    label = data[1]

    features = model(imgs)
    label = torch.ones((label.shape[0]))
    tsne_array.append(features.cpu().detach())
    label_array.append(label.cpu().detach())

tsne_data = torch.cat(tsne_array, 0).numpy()
label_data = torch.cat(label_array, 0).numpy()
print(label_data.shape)
x_tsne = manifold.TSNE(n_components=2, init='random', random_state=5, verbose=1).fit_transform(tsne_data)
x_min, x_max = x_tsne.min(0), x_tsne.max(0)

X_norm = (x_tsne - x_min) / (x_max - x_min) 
# print(X_norm.shape)
# print(label.shape)


for i in range(X_norm.shape[0]):
        plt.text(X_norm[i, 0], X_norm[i, 1], str(int(label_data[i])), color=plt.cm.Set3(int(label_data[i])),
            fontdict={'weight': 'bold', 'size': 9})



plt.xticks([])
plt.yticks([])
plt.savefig("tsne_mnist_usps_domain.png")

