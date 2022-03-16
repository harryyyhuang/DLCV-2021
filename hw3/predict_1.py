import argparse
from unicodedata import name
from pytorch_pretrained_vit import ViT
from data_loader import build_data
from torch.utils.data import DataLoader
import torch
import numpy as np 
from tqdm import tqdm
import os
import csv

def write_p1(args, name_list, label_list):
    with open(args.predict_path, 'w') as csvfile:
        csv_writer = csv.writer(csvfile)
        field = ['filename', 'label']
        csv_writer.writerow(field)

        for (name, label) in zip(name_list, label_list):
            csv_writer.writerow([name[0], label.cpu().numpy()[0]])

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

parser = argparse.ArgumentParser(
    description="training vit parameter")
parser.add_argument("--p1_path", default="./hw3_data/p1_data/val", type=str)
parser.add_argument("--batch_size", default=1, type=int)
parser.add_argument("--epoch", default=1, type=int)
parser.add_argument("--dataset", default="p1")
parser.add_argument("--predict_path", default="./predict.csv", type=str)
args = parser.parse_args()


same_seeds(714)
device  = get_device()
print(device)


val_dataset = build_data(args, "test")
val_loader = DataLoader(val_dataset, batch_size = args.batch_size, shuffle = False, num_workers = 8, pin_memory = True)
model = ViT('B_16_imagenet1k',image_size=384, pretrained=True, num_classes=37).to(device)
model.device = device
model.load_state_dict(torch.load("p1.pt"))


for epoch in range(args.epoch):

    model.eval()

    valid_label = []
    valid_name = []

    for batch in tqdm(val_loader):

        # A batch consists of image data and corresponding labels.
        image, label, data_name = batch

        # We don't need gradient in validation.
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
          logits, _ = model(image.to(device))

        out = logits.argmax(dim=1)

        valid_name.append(data_name)
        valid_label.append(out)

    write_p1(args, valid_name, valid_label)

      
