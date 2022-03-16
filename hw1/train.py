import numpy as np
from torch.utils.data import DataLoader
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
from mean_iou_evaluate import my_mean_iou_score
from PIL import Image

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
parser.add_argument("--batch_size", default=8, type=int)
parser.add_argument("--epoch", default=50, type=int)
parser.add_argument("--model_path", default="./model_pth", type=str)
parser.add_argument("--p2_out_path", default="./train/predict", type=str)
args = parser.parse_args()

# get device
device = get_device()
print(f'DEVICE: {device}')

train_dataset = build_data(args, "train")
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
val_dataset = build_data(args, "val")
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)


model = build_model(args)

model = model.to(device)
model.device = device


if(args.dataset == "p1"):
  optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
else:
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
class_weight = torch.tensor([[2.0, 1.0, 3.0, 1.0, 2.0, 3.0, 1.0]]).to(device)
last_valid_score = 0

if(args.dataset == "p1"):
  criterion = nn.CrossEntropyLoss()
else:
  criterion = nn.CrossEntropyLoss(weight=class_weight)

cls_color = {
    0:  [0, 255, 255],
    1:  [255, 255, 0],
    2:  [255, 0, 255],
    3:  [0, 255, 0],
    4:  [0, 0, 255],
    5:  [255, 255, 255],
    6: [0, 0, 0],
}

for epoch in range(args.epoch):
    print("in epoch {}".format(epoch))
    model.train()


    train_loss = []
    train_accs = []

    for (imgs, labels, name) in train_loader:

        optimizer.zero_grad()
        logits = model(imgs.to(device))
        loss = criterion(logits, labels.to(device))

        loss.backward()
        optimizer.step()

        if(args.dataset == "p2"):
          out = logits.argmax(dim=1)
          labels = labels.to(device)
          acc = my_mean_iou_score(out.cpu().numpy(), labels.cpu().numpy())
        else:
          out = logits.argmax(dim=-1)
          acc = (out == labels.to(device)).float().mean()

        train_loss.append(loss.item())
        train_accs.append(acc)


    # The average loss and accuracy of the training set is the average of the recorded values.
    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_accs) / len(train_accs)

    # experiment.log_metric("train_loss", train_loss, step=epoch)
    # experiment.log_metric("train_acc", train_acc, step=epoch)


    # Print the information.
    print(f"[ Train | {epoch + 1:03d}/{args.epoch:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")


    
    model.eval()

    valid_loss = []
    valid_accs = []

    for (imgs, labels, name) in val_loader:

        with torch.no_grad():
          logits = model(imgs.to(device))


        loss = criterion(logits, labels.to(device))

        if(args.dataset == "p2"):
          out = logits.argmax(dim=1)
          labels = labels.to(device)
          acc = my_mean_iou_score(out.cpu().numpy(), labels.cpu().numpy(), True)
          out = out.cpu().numpy()
        else:
          out = logits.argmax(dim=1)
          acc = (out == labels.to(device)).float().mean()

        if(args.dataset == "p2"):
          if(epoch == 18 or epoch == 34 or epoch == 49):
            # print(name)
            if(name[0] == "0010_mask.png" or name[0] == "0097_mask.png" or name[0] == "0107_mask.png"):
              image_numpy = np.zeros((out.shape[1], out.shape[2], 3))
              for i in range(7):
                image_numpy[(out == i).squeeze(0)] = cls_color[i]
              out_mask = Image.fromarray(image_numpy.astype(np.uint8), 'RGB')
              out_mask.save(os.path.join(args.p2_out_path, str(epoch)+"_"+name[0]))
            



        # Record the loss and accuracy.
        valid_loss.append(loss.item())
        valid_accs.append(acc)


    


    # The average loss and accuracy for entire validation set is the average of the recorded values.
    valid_loss = sum(valid_loss) / len(valid_loss)
    valid_acc = sum(valid_accs) / len(valid_accs)




    # Print the information.
    print(f"[ Valid | {epoch + 1:03d}/{args.epoch:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}, currently best acc = {last_valid_score}")

    if(valid_acc > last_valid_score):
      print("saveing the model for prob {}...".format(args.dataset))
      torch.save(model.state_dict(), os.path.join("model_pth", args.dataset+".pt"))
      last_valid_score = valid_acc

