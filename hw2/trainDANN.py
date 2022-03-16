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
same_seeds(116)
parser = argparse.ArgumentParser(
description="training parameter")
parser.add_argument("--p1_path", default="./hw2_data/face", type=str)
parser.add_argument("--p1_test_path", default="./hw2_data/face/test", type=str)
parser.add_argument("--p1_gen_path", default="./gen", type=str)
parser.add_argument("--p2_gen_path", default="./gen_p2", type=str)
parser.add_argument("--p2_path", default="./hw2_data/digits/", type=str)
parser.add_argument("--dataset", default="p3")
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--model", default="p3")
parser.add_argument("--epoch", default=1, type=int)
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
lr = 0.001

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

# Model
model = build_model(args)
model.to(device)
model.eval()

# Loss
criterion = nn.CrossEntropyLoss()

""" Medium: Use RMSprop for WGAN. """
# Optimizer
opt_model = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))

# opt_D = torch.optim.RMSprop(D.parameters(), lr=lr)
# opt_G = torch.optim.RMSprop(G.parameters(), lr=lr)

# DataLoader
train_dataset, test_dataset = build_data(args, "usps2svhn")
true_test = build_data(args, "svhn")
dataloader_train = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
dataloader_test = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
dataloader_treu_test = DataLoader(true_test, batch_size=args.batch_size, shuffle=True, num_workers=2)

steps = 0
domain_acc = 0
for e, epoch in enumerate(range(n_epoch)):
    # progress_bar = qqdm(dataloader_train)
    accs_class = []
    lens = []
    # len_dataloader = min(len(dataloader_train), len(dataloader_test))
    # data_source_iter = iter(dataloader_train)
    # data_domain_iter =  iter(dataloader_test)


    # model.train()
    # model.domainClassifier.eval()
    # for i in tqdm(range(len_dataloader)):


    #     p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
    #     alpha = 3. / (1. + np.exp(-10 * p)) - 1

    #     data_source = data_source_iter.next()
    #     imgs = data_source[0].cuda()
    #     labels = data_source[1].cuda()

    #     model.zero_grad()

    #     cla_logit, domain_logit = model(imgs, alpha)
    #     loss_cls = criterion(cla_logit, labels.type(torch.long))
    #     loss_domain = criterion(domain_logit, torch.zeros(len(labels)).cuda().type(torch.long))


    #     data_target = data_domain_iter.next()
    #     t_imgs = data_target[0].cuda()
    #     t_labels = data_target[1].cuda()

    #     cls_logit, domain_logit = model(t_imgs, alpha)
    #     # loss_cls_target = criterion(cls_logit, t_labels.type(torch.long))
    #     loss_t_domain = criterion(domain_logit, torch.ones(len(t_labels)).cuda().type(torch.long))
    #     loss = loss_cls + loss_domain + loss_t_domain
    #     # loss = loss_cls
    #     loss.backward()
    #     opt_model.step()


    model.eval()
    for data in tqdm(dataloader_treu_test):
        imgs, label = data
        imgs = imgs.cuda()
        label = label.cuda()
        
        cls_logit, domain_logit = model(imgs)
        domain_class_out = cls_logit.argmax(dim=1)
        acc_class = (domain_class_out == label).float().mean()
        accs_class.append(acc_class)



    tmp_acc  = sum(accs_class)/len(accs_class)
    print(f"[ Train | {epoch + 1:03d}/{args.epoch:03d} ] , acc = {tmp_acc:.5f}, current_best = {domain_acc:.5f}")
    if(tmp_acc > domain_acc):
        domain_acc = tmp_acc
        print("saveing the model for prob {}...".format(args.dataset))
        # torch.save(model.state_dict(), os.path.join(ckpt_dir, 'DANNEASY.pth'))
    