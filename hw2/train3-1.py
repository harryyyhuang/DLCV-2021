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
same_seeds(52)
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
parser.add_argument("--epoch", default=500, type=int)
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
lr = 0.00001

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
model.train()

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
    progress_bar = qqdm(dataloader_train)
    accs_class = []
    lens = []
    len_dataloader = min(len(dataloader_train), len(dataloader_test))
    data_source_iter = iter(dataloader_train)
    data_domain_iter =  iter(dataloader_test)


    model.train()
    for i in tqdm(range(len_dataloader)):


        p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        data_source = data_source_iter.next()
        imgs = data_source[0].cuda()
        labels = data_source[1].cuda()

        model.zero_grad()

        cla_logit, domain_logit = model(imgs, alpha)
        loss_cls = criterion(cla_logit, labels.type(torch.long))
        loss_domain = criterion(domain_logit, torch.zeros(len(labels)).cuda().type(torch.long))


        data_target = data_domain_iter.next()
        t_imgs = data_target[0].cuda()
        t_labels = data_target[1].cuda()

        cls_logit, domain_logit = model(t_imgs, alpha)
        loss_t_domain = criterion(domain_logit, torch.ones(len(t_labels)).cuda().type(torch.long))
        loss = loss_cls + loss_domain + loss_t_domain
        loss.backward()
        opt_model.step()


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
        torch.save(model.state_dict(), os.path.join(ckpt_dir, 'DANNUS_.pth'))
    #     FeatureM.train()
    #     extractedFeature = FeatureM(imgs)
    #     train_index = (datatype == 0)
    #     test_index = (datatype == 1)
    #     # print(imgs)
    #     # print(labels
    #     train_feature = extractedFeature[train_index]
    #     test_feature = extractedFeature[test_index]

    #     ClassM.train()
    #     train_class_logit = ClassM(train_feature)
    #     loss_class = criterion(train_class_logit, labels[train_index].type(torch.long))
    #     FeatureM.zero_grad()
    #     ClassM.zero_grad()
    #     loss_class.backward()
    #     opt_Class.step()
    #     opt_Feature.step()


    #     ClassM.eval()
    #     domain_class_logit = ClassM(test_feature)

    #     domain_class_out = domain_class_logit.argmax(dim=1)
    #     acc_class = (domain_class_out == labels[test_index]).float().sum()
    #     accs_class.append(acc_class)
    #     lens.append(domain_class_out.shape[0])


        
    #     extractedFeature = FeatureM(imgs)
    #     DomainM.train()
    #     train_domain_logit = DomainM(extractedFeature)

    #     loss_domain = criterion(train_domain_logit, datatype.type(torch.long))
    #     DomainM.zero_grad()
    #     loss_domain.backward()
    #     opt_Domain.step()
    #     FeatureM.zero_grad()
    #     loss_domain = -loss_domain
    #     loss_domain.backward()
    #     opt_Feature.step()


    #     train_domain_logit = train_domain_logit.argmax(dim=1)
    #     acc_domain = (train_domain_logit == datatype).float().sum()
    #     accs_domain.append(acc_domain)


    #     steps += 1
    #     # print(steps)
    #     # Set the info of the progress bar
    #     #   Note that the value of the GAN loss is not directly related to
    #     #   the quality of the generated images.
        
    #     progress_bar.set_infos({
    #         'ACC_Class': round(sum(acc_class)/sum(lens), 4),
    #         'ACC_Domain': round(sum(accs_domain)/len(accs_domain), 4),
    #         'Epoch': e+1,
    #         'Step': steps,
    #     })

    # train_acc = sum(accs) / len(accs)
    # print(f"[ Train | {epoch + 1:03d}/{args.epoch:03d} ] , acc = {train_acc:.5f}")






    

    # # if (e+1) % 5 == 0 or e == 0:

    # G.eval()
    # D.eval()

    # accs = []


    # # f_imgs_sample = G(z_sample)
    # # print(f_imgs.shape)
    # # exit()

    # total_num = 1
    # for digit_num in range(10):
    #     for k in range(test_len//(args.batch_size)+1):
    #         z_sample = Variable(torch.randn(args.batch_size, z_dim)).cuda()
    #         f_imgs_sample = (G(z_sample, torch.ones(args.batch_size).cuda()*digit_num).data + 1) / 2.0
    #         if (k != (test_len//(args.batch_size))):
    #             t_range = args.batch_size
    #         else:
    #             t_range = test_len % args.batch_size

    #         pred_logits = net(f_imgs_sample)
    #         out = pred_logits.argmax(dim=1)
    #         acc = (out == torch.ones(args.batch_size).cuda()*digit_num).float().mean()
    #         accs.append(acc)
            



    #         for t in range(t_range):
    #             filename = os.path.join(args.p2_gen_path, f'{int(digit_num)}_{str(total_num).zfill(3)}.png')
    #             torchvision.utils.save_image(f_imgs_sample[t], filename) 
    #             # print(f' | Save samples to {filename}.')
    #             total_num+=1
    #     total_num-=100

    # valid_acc = sum(accs) / len(accs)
    # print(f"[ Valid | {epoch + 1:03d}/{args.epoch:03d} ] , acc = {valid_acc:.5f}, currently best acc = {last_valid_score}")

    # if(valid_acc > last_valid_score):
    #     print("saveing the model for prob {}...".format(args.dataset))
    #     torch.save(G.state_dict(), os.path.join(ckpt_dir, 'G_p2.pth'))
    #     torch.save(D.state_dict(), os.path.join(ckpt_dir, 'D_p2.pth'))
    #     last_valid_score = valid_acc