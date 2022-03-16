# import comet_ml at the top of your file
# from comet_ml import Experiment
import numpy as np
from torch.utils.data import DataLoader
import gc
from data_loader import build_data
from model import build_model
import torch
import torch.nn as nn
import argparse
import math
from torch.utils.data import ConcatDataset, DataLoader, Subset, random_split
from torch.optim.lr_scheduler import LambdaLR
import os

import torchvision.models as models
from torchinfo import summary
from tqdm import tqdm
from mean_iou_evaluate import my_mean_iou_score
from torchvision.utils import save_image
from PIL import Image

# # Create an experiment with your api key
# experiment = Experiment(
#     api_key="gbIbPvPkSfvA1rSVHFO2B1qAD",
#     project_name="dlcv-hw1",
#     workspace="harryyyhuang",
# )



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
parser.add_argument("--epoch", default=1000, type=int)
parser.add_argument("--warmup_step", default=1000, type=int)
parser.add_argument("--model_path", default="./model_pth/p2.pt", type=str)
parser.add_argument("--predict_path", default="./predict", type=str)
args = parser.parse_args()

# get device
device = get_device()
print(f'DEVICE: {device}')

train_dataset = build_data(args, "train")
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
val_dataset = build_data(args, "val")
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

EARLY_STOP = 50


model = build_model(args)

model = model.to(device)
model.device = device



optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
class_weight = torch.tensor([[2.0, 1.0, 3.0, 1.0, 2.0, 3.0, 1.0]]).to(device)
last_valid_score = 0

criterion = nn.CrossEntropyLoss(weight = class_weight)

model.load_state_dict(torch.load(args.model_path))

cls_color = {
    0:  [0, 255, 255],
    1:  [255, 255, 0],
    2:  [255, 0, 255],
    3:  [0, 255, 0],
    4:  [0, 0, 255],
    5:  [255, 255, 255],
    6: [0, 0, 0],
}

for epoch in range(1):

    
    model.eval()

    valid_loss = []
    valid_accs = []

    for (imgs, labels, name) in tqdm(val_loader):
        # imgs = imgs.view(-1, 3, imgs.shape[3], imgs.shape[4])

        # labels = labels.view(-1, labels.shape[2], labels.shape[3])

        with torch.no_grad():
          logits = model(imgs.to(device))


        loss = criterion(logits, labels.to(device))

        if(args.dataset == "p2"):
          out = logits.argmax(dim=1).cpu().numpy()
          labels = labels.to(device)
        #   print(out.shape)
          image_numpy = np.zeros((out.shape[1], out.shape[2], 3))
        #   print((out == 1).shape)
        #   exit()
          for i in range(7):
              image_numpy[(out == i).squeeze(0)] = cls_color[i]
        #   out_mask = Image.fromarray(out.astype(np.uint8))
          out_mask = Image.fromarray(image_numpy.astype(np.uint8), 'RGB')
          out_mask.save(os.path.join("predict", name[0]))
        #   acc = my_mean_iou_score(out, labels, True)
        else:
          out = logits.argmax(dim=1)
          acc = (out == labels.to(device)).float().mean()




