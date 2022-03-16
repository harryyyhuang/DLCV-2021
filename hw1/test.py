import numpy as np
from torch.utils.data import DataLoader
from data_loader import build_data
from model import build_model
import torch
import torch.nn as nn
import argparse
import math
from torch.utils.data import ConcatDataset, DataLoader, Subset, random_split
import os

import torchvision.models as models
from mean_iou_evaluate import my_mean_iou_score
from torchvision.utils import save_image
from PIL import Image
import csv


def write_p1(args, name_list, label_list):
    with open(args.p1_out_path, 'w') as csvfile:
        csv_writer = csv.writer(csvfile)
        field = ['image_id', 'label']
        csv_writer.writerow(field)

        for (name, label) in zip(name_list, label_list):
            csv_writer.writerow([name[0], label.cpu().numpy()[0]])

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
parser.add_argument("--p1_path", default="./hw1_data/p1_data", type=str)
parser.add_argument("--p2_path", default="./hw1_data/p2_data", type=str)
parser.add_argument("--dataset", default="p2")
parser.add_argument("--batch_size", default=1, type=int)
parser.add_argument("--p1_out_path", default="./output/pred.csv", type=str)
parser.add_argument("--p2_out_path", default="./predict", type=str)
args = parser.parse_args()

# get device
device = get_device()
print(f'DEVICE: {device}')

val_dataset = build_data(args, "test")
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)



model = build_model(args)

model = model.to(device)
model.device = device

if(args.dataset == 'p1'):
    model.load_state_dict(torch.load("p1_strong.pt"))
else:
    model.load_state_dict(torch.load("p2_strong.pt"))

cls_color = {
    0:  [0, 255, 255],
    1:  [255, 255, 0],
    2:  [255, 0, 255],
    3:  [0, 255, 0],
    4:  [0, 0, 255],
    5:  [255, 255, 255],
    6: [0, 0, 0],
}

model.eval()
# print(model)
valid_loss = []
valid_accs = []
if args.dataset == "p1":
    valid_name = []
    valid_label = []
print("start")
for (imgs, labels, name) in val_loader:

    with torch.no_grad():
        logits = model(imgs.to(device))

    

    if(args.dataset == "p2"):
        out = logits.argmax(dim=1).cpu().numpy()
        labels = labels.to(device)
        image_numpy = np.zeros((out.shape[1], out.shape[2], 3))
        for i in range(7):
            image_numpy[(out == i).squeeze(0)] = cls_color[i]
        out_mask = Image.fromarray(image_numpy.astype(np.uint8), 'RGB')
        out_mask.save(os.path.join(args.p2_out_path, name[0]))

    else:
        out = logits.argmax(dim=1)
        acc = (out == labels.to(device)).float().mean()
        valid_accs.append(acc)

    if args.dataset == "p1":
        valid_name.append(name)
        valid_label.append(out)



if(args.dataset == "p1"):
    write_p1(args, valid_name, valid_label)
    valid_acc = sum(valid_accs) / len(valid_accs)
    print(valid_acc)





