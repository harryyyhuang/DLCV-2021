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
import torch.nn.functional as F

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
parser.add_argument("--p2_path", default="./hw2_data/digits/mnistm", type=str)
parser.add_argument("--dataset", default="p2")
parser.add_argument("--batch_size", default=256, type=int)
parser.add_argument("--model", default="p2")
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

# Model
G, D = build_model(args)
G.train()
D.train()

# Loss
criterion = nn.CrossEntropyLoss()

""" Medium: Use RMSprop for WGAN. """
# Optimizer
opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.9, 0.999))
opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.9, 0.999))
# opt_D = torch.optim.RMSprop(D.parameters(), lr=lr)
# opt_G = torch.optim.RMSprop(G.parameters(), lr=lr)

# best_fid = 10000
# best_is = 0

# def load_checkpoint(checkpoint_path, model):
#     state = torch.load(checkpoint_path, map_location = "cuda")
#     model.load_state_dict(state['state_dict'])
#     print('model loaded from %s' % checkpoint_path)
# class Classifier(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 4 * 4, 128)
#         self.fc2 = nn.Linear(128, 64)
#         self.fc3 = nn.Linear(64, 10)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = torch.flatten(x, 1) # flatten all dimensions except batch
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

# DataLoader
# train_dataset = build_data(args, "train")
# dataloader_train = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
# net = Classifier()
# path = "Classifier.pth"
# load_checkpoint(path, net)
# net.eval()
# use_cuda = torch.cuda.is_available()
# device = torch.device("cuda" if use_cuda else "cpu")
# print('Device used:', device)
# if torch.cuda.is_available():
#     net = net.to(device)
# last_valid_score = 0

# steps = 0
for e, epoch in enumerate(range(n_epoch)):
#     progress_bar = qqdm(dataloader_train)
    # for i, data in enumerate(progress_bar):
    #     imgs = data[0].cuda()
    #     labels = data[1].cuda()

    #     bs = imgs.size(0)

    #     # ============================================
    #     #  Train D
    #     # ============================================
    #     z = Variable(torch.randn(bs, z_dim)).cuda()
    #     # print(z.shape)
    #     r_imgs = Variable(imgs).cuda()
    #     f_imgs = G(z, labels)

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
    #     _, D_inter_img= D(inter_img)
    #     gradients = torch_grad(outputs=D_inter_img, inputs=inter_img,
    #                                 grad_outputs=torch.ones(D_inter_img.size()).cuda(),
    #                                 create_graph=True, retain_graph=True)[0]
    #     gradients = gradients.view(bs, -1)
    #     gradient_penalty = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
    #     gradient_penalty =  lambda_norm * ((gradient_penalty - 1) ** 2).mean()

    #     out_R_L, out_R_D =  D(r_imgs)
    #     out_F_L, out_F_D =  D(f_imgs)

    #     # class
    #     loss_R_L = criterion(out_R_L, labels.type(torch.long))
    #     # loss_F_L = criterion(out_F_L, labels.type(torch.long))
        
    #     # WGAN Loss
        
    #     loss_D = -torch.mean(out_R_D) + torch.mean(out_F_D) + gradient_penalty + loss_R_L 
       
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
    #         z = Variable(torch.randn(bs, z_dim)).cuda()
    #         f_imgs = G(z, labels)

    #         # Model forwarding
    #         # f_logit = D(f_imgs)
            
    #         """ Medium: Use WGAN Loss"""
    #         # Compute the loss for the generator.
    #         # loss_G = criterion(f_logit, r_label)
    #         out_F_L, out_F_D =  D(f_imgs)
    #         loss_F_L = criterion(out_F_L, labels.type(torch.long))
    #         # WGAN Loss
    #         # print(out_F_D)
            
    #         loss_G = -torch.mean(out_F_D) + loss_F_L

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




    

    # if (e+1) % 5 == 0 or e == 0:

    G.eval()
    D.eval()

    accs = []


    # f_imgs_sample = G(z_sample)
    # print(f_imgs.shape)
    # exit()

    save_img = torch.zeros((10, 10, 3, 28, 28)) 

    total_num = 1
    for digit_num in range(10):
        for k in range(test_len//(args.batch_size)+1):
            z_sample = Variable(torch.randn(args.batch_size, z_dim)).cuda()
            f_imgs_sample = (G(z_sample, torch.ones(args.batch_size).cuda()*digit_num).data + 1) / 2.0
            if (k != (test_len//(args.batch_size))):
                t_range = args.batch_size
            else:
                t_range = test_len % args.batch_size

            # pred_logits = net(f_imgs_sample)
            # out = pred_logits.argmax(dim=1)
            # acc = (out == torch.ones(args.batch_size).cuda()*digit_num).float().mean()
            # accs.append(acc)
            



            for t in range(t_range):
                filename = os.path.join(args.p2_gen_path, f'{int(digit_num)}_{str(total_num).zfill(3)}.png')
                torchvision.utils.save_image(f_imgs_sample[t], filename) 
                # print(f' | Save samples to {filename}.')
                total_num+=1
            if(k == 0):
                # save_img[10*digit_num:10*digit_num+10] = f_imgs_sample[:10]
                save_img[digit_num, :, :, :] = f_imgs_sample[:10]
        total_num-=100

    # torchvision.utils.save_image(save_img.permute(1,0,2,3,4).reshape(-1, 3, 28, 28), "p2.png", nrow=10)

    # valid_acc = sum(accs) / len(accs)
    # print(f"[ Valid | {epoch + 1:03d}/{args.epoch:03d} ] , acc = {valid_acc:.5f}, currently best acc = {last_valid_score}")

    # if(valid_acc > last_valid_score):
    #     print("saveing the model for prob {}...".format(args.dataset))
    #     torch.save(G.state_dict(), os.path.join(ckpt_dir, 'G_p2.pth'))
    #     torch.save(D.state_dict(), os.path.join(ckpt_dir, 'D_p2.pth'))
    #     last_valid_score = valid_acc