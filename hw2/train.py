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
import torchvision.models as models
import torchvision
from PIL import Image
from torch.autograd import grad as torch_grad
import matplotlib.pyplot as plt

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
parser.add_argument("--p2_path", default="./hw2_data/digits", type=str)
parser.add_argument("--dataset", default="p1")
parser.add_argument("--batch_size", default=512, type=int)
parser.add_argument("--model", default="p1")
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
z_dim = 128
# z_sample = Variable(torch.randn(len(os.listdir(args.p1_test_path)), z_dim, 1 ,1)).cuda()
lr = 0.00002

""" Medium: WGAN, 50 epoch, n_critic=5, clip_value=0.01 """
n_epoch = args.epoch # 50
n_critic = 5 # 5
clip_value = 0.01
lambda_norm = 10.0
# test_len = len(os.listdir(args.p1_test_path))
log_dir = os.path.join(workspace_dir, 'logs')
ckpt_dir = os.path.join(workspace_dir, 'checkpoints')
os.makedirs(log_dir, exist_ok=True)
os.makedirs(ckpt_dir, exist_ok=True)

# Model
G, D = build_model(args)
G.train()
D.train()

# Loss
criterion = nn.BCELoss()

""" Medium: Use RMSprop for WGAN. """
# Optimizer
opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
# opt_D = torch.optim.RMSprop(D.parameters(), lr=lr)
# opt_G = torch.optim.RMSprop(G.parameters(), lr=lr)

best_fid = 10000
best_is = 0

# DataLoader
# train_dataset = build_data(args, "train")
# dataloader_train = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
# steps = 0
for e, epoch in enumerate(range(n_epoch)):
    # progress_bar = qqdm(dataloader_train)
    # for i, data in enumerate(progress_bar):
    #     imgs = data
    #     imgs = imgs.cuda()

    #     bs = imgs.size(0)

    #     # ============================================
    #     #  Train D
    #     # ============================================
    #     z = Variable(torch.randn(bs, z_dim, 1 ,1)).cuda()
    #     # print(z.shape)
    #     r_imgs = Variable(imgs).cuda()
    #     f_imgs = G(z)

    #     """ Medium: Use WGAN Loss. """
    #     # Label
    #     # r_label = torch.ones((bs)).cuda()
    #     # f_label = torch.zeros((bs)).cuda()

    #     # Model forwarding
    #     # r_logit = D(r_imgs.detach())
    #     # f_logit = D(f_imgs.detach())
        
    #     # Compute the loss for the discriminator.
    #     # r_loss = criterion(r_logit, r_label)
    #     # f_loss = criterion(f_logit, f_label)
    #     # loss_D = (r_loss + f_loss) / 2
    #     noise = Variable(torch.rand(bs, 1, 1, 1).cuda())
    #     noise = noise.expand_as(r_imgs)
    #     inter_img = Variable((noise*r_imgs.data + (1-noise)*f_imgs.data), requires_grad=True).cuda()
    #     D_inter_img = D(inter_img)
    #     gradients = torch_grad(outputs=D_inter_img, inputs=inter_img,
    #                                 grad_outputs=torch.ones(D_inter_img.size()).cuda(),
    #                                 create_graph=True, retain_graph=True)[0]
    #     gradients = gradients.view(bs, -1)
    #     gradient_penalty = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
    #     gradient_penalty =  lambda_norm * ((gradient_penalty - 1) ** 2).mean()
        
    #     # WGAN Loss
    #     loss_D = -torch.mean(D(r_imgs)) + torch.mean(D(f_imgs)) + gradient_penalty
       
    #     # loss_D = -torch.mean(D(r_imgs)) + torch.mean(D(f_imgs))
    #     # Model backwarding
    #     D.zero_grad()
    #     loss_D.backward()

    #     # Update the discriminator.
    #     opt_D.step()
        
    #     # # """ Medium: Clip weights of discriminator. """
    #     # for p in D.parameters():
    #     #    p.data.clamp_(-clip_value, clip_value)

    #     # ============================================
    #     #  Train G
    #     # ============================================
    #     if steps % n_critic == 0:
    #         # Generate some fake images.
    #         z = Variable(torch.randn(bs, z_dim, 1, 1)).cuda()
    #         f_imgs = G(z)

    #         # Model forwarding
    #         # f_logit = D(f_imgs)
            
    #         """ Medium: Use WGAN Loss"""
    #         # Compute the loss for the generator.
    #         # loss_G = criterion(f_logit, r_label)
    #         # WGAN Loss
    #         loss_G = -torch.mean(D(f_imgs))

    #         # Model backwarding
    #         G.zero_grad()
    #         loss_G.backward()

    #         # Update the generator.
    #         opt_G.step()

    #     steps += 1
    #     # print(steps)
    #     # Set the info of the progress bar
    #     #   Note that the value of the GAN loss is not directly related to
    #     #   the quality of the generated images.
    #     progress_bar.set_infos({
    #         'Loss_D': round(loss_D.item(), 4),
    #         'Loss_G': round(loss_G.item(), 4),
    #         'Epoch': e+1,
    #         'Step': steps,
    #     })
    #     # break



    

    # if (e+1) % 5 == 0 or e == 0:

    G.eval()
    D.eval()

    # f_imgs_sample = G(z_sample)
    # print(f_imgs.shape)
    # exit()

    total = 0
    for k in range(1000//(64)+1):
        z_sample = Variable(torch.randn(64, z_dim, 1, 1)).cuda()
        f_imgs_sample = (G(z_sample).data + 1) / 2.0
        if (k != (1000//(64))):
            t_range = 64
        else:
            t_range = 1000 % 64

        for t in range(t_range):
        # for k in range(test_len):
            filename = os.path.join(args.p1_gen_path, f'{str(k*64+t+1).zfill(4)}.png')
            torchvision.utils.save_image(f_imgs_sample[t], filename) 
            print(f' | Save samples to {filename}.')

    # # # exit()
    # fid_value = calculate_fid_given_paths([args.p1_gen_path, args.p1_test_path],
    #                                         32,
    #                                         device,
    #                                         2048,
    #                                         2)

    # test_dataset = build_data(args, "gen")
    # # dataloader_train = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    # is_valie = inception_score(test_dataset, cuda=True, batch_size=32)[0]

    # print(f'The FID in epoch {e} is {fid_value}')
    # print(f'The IS in epoch {e} is {is_valie}')
    # G.train()
    # D.train()

    # if(is_valie > best_is):
    #     best_is = is_valie
    #     best_fid = fid_value
    #     # Save the checkpoints.
    #     torch.save(G.state_dict(), os.path.join(ckpt_dir, 'G.pth'))
    #     torch.save(D.state_dict(), os.path.join(ckpt_dir, 'D.pth'))
    # if(fid_value < best_fid):
    #     best_fid = fid_value
    #     best_is = is_valie
    #     # Save the checkpoints.
    #     torch.save(G.state_dict(), os.path.join(ckpt_dir, 'G.pth'))
    #     torch.save(D.state_dict(), os.path.join(ckpt_dir, 'D.pth'))

    # print(f'Epoch {e} best fid is {best_fid} best is is {best_is}')
        



