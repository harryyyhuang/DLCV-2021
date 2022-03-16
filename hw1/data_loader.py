from posixpath import split
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
import os
from PIL import Image
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import imageio

train_tfm_p1 = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomAffine(30),
    # transforms.RandomRotation(degrees=(-90, 90)),
    transforms.ColorJitter(0.5, 0.5, 0.5, 0.5),
    # transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    # transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
])
# train_tfm_p1 = transforms.RandomApply(train_tfm_p1, p = 0.5)

val_tfm_p1 = transforms.Compose([
    # transforms.RandomAffine(30),
    # transforms.RandomRotation(degrees=(-90, 90)),
    # transforms.ColorJitter(0.5, 0.5, 0.5, 0.5),
    # transforms.RandomCrop(32),
    # transforms.RandomHorizontalFlip(p=0.5),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
])

def crop(image, target, region):
    cropped_image = F.crop(image, *region)

    # target = target.copy()
    i, j, h, w = region
    target = target[i:i+h, j:j+w]

    return cropped_image, target

class RandomCrop(object):
    def __init__(self, size):
        self.size = size
    def __call__(self, img, target):
        region = transforms.RandomCrop.get_params(img, self.size)
        return crop(img, target, region)


def hflip(image, target):
    flipped_image = F.hflip(image)

    target = target.flip(-1)

    return flipped_image, target

class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, img, target):
        if random.random() < self.p:
            return hflip(image, target)
        return img, target

def vflip(image, target):
    flipped_image = F.vflip(image)

    target = target.flip(-2)

    return flipped_image, target

class RandomVeticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, img, target):
        if random.random() < self.p:
            return vflip(image, target)
        return img, target

train_p2 = transforms.Compose([
    # transforms.RandomAffine(30),
    # transforms.RandomRotation(degrees=(-50, 50)),
    # transforms.ColorJitter(5, 5, 5),
    # transforms.RandomCrop(32),
    # transforms.RandomHorizontalFlip(p=0.5),
    # transforms.RandomVerticalFlip(p=0.5),
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
])

valid_p2 = transforms.Compose([
    # transforms.RandomAffine(30),
    # transforms.RandomRotation(degrees=(-50, 50)),
    # transforms.ColorJitter(5, 5, 5),
    # transforms.RandomCrop(32),
    # transforms.RandomHorizontalFlip(p=0.5),
    # transforms.RandomVerticalFlip(p=0.5),
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
])




class TIMITDatasetP1(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.files_name = sorted(sorted(os.listdir(data_dir), key = lambda x: int(x.split('_')[1].split('.')[0])),  key = lambda x: int(x.split('_')[0]))
        self.transform = transform

    def __getitem__(self, idx):
        data_name = self.files_name[idx]
        image = Image.open(os.path.join(self.data_dir, data_name))
        split_name = data_name.split("_")
        label = int(split_name[0])

        if self.transform is not None:
            image = self.transform(image)
        return image, label, data_name

    def __len__(self):
        return len(self.files_name)

class TIMITDatasetP2(Dataset):
    def __init__(self, data_dir,):
        self.data_dir = data_dir
        self.sat_name = []
        self.mask_name = []

        for file in os.listdir(data_dir):
            if("mask" in file):
                self.mask_name.append(file)
            else:
                self.sat_name.append(file)

        self.sat_name = sorted(self.sat_name)
        self.mask_name = sorted(self.mask_name)

    def read_mask(self, filepath):
        mask = imageio.imread(filepath)
        mask = (mask >= 128).astype(int)

        mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]

        mask_convert = torch.zeros((mask.shape[0], mask.shape[1]), dtype=torch.long)

        mask_convert[mask == 3] = 0  # (Cyan: 011) Urban land 
        mask_convert[mask == 6] = 1  # (Yellow: 110) Agriculture land 
        mask_convert[mask == 5] = 2  # (Purple: 101) Rangeland 
        mask_convert[mask == 2] = 3  # (Green: 010) Forest land 
        mask_convert[mask == 1] = 4  # (Blue: 001) Water 
        mask_convert[mask == 7] = 5  # (White: 111) Barren land 
        mask_convert[mask == 0] = 6  # (Black: 000) Unknown 

        return mask_convert


    def __getitem__(self, idx):
        data_name = self.sat_name[idx]
        mask_name = self.mask_name[idx]
        image = Image.open(os.path.join(self.data_dir, data_name))
        mask = self.read_mask(os.path.join(self.data_dir, mask_name))
        

        # if(self.data_dir == 'train'):
        #     # thecrop = RandomCrop((256, 256))
        #     # image, mask = thecrop(image, mask)
        #     theflip = RandomHorizontalFlip()
        #     image, mask = theflip(image, mask)
        #     thevlip = RandomVerticalFlip()
        #     image, mask = thevlip(image, mask)


        image = train_p2(image)
        # else:
        #     image = valid_p2(image)
        # oneset_img = torch.ones((4,3,256,256))
        # oneset_mask = torch.ones((4,256,256), dtype=torch.long)
        # imag_1 = image[:, :256, :256]
        # mask_1 = mask[:256, :256]
        # #print(imag_1.shape)
        # oneset_img[0] = imag_1
        # oneset_mask[0] = mask_1
        # imag_2 = image[:, 256:, :256]
        # mask_2 = mask[256:, :256]
        # oneset_img[1] = imag_2
        # oneset_mask[1] = mask_2
        # imag_3 = image[:, 256:, 256:]
        # mask_3 = mask[256:, 256:]
        # oneset_img[2] = imag_3
        # oneset_mask[2] = mask_3
        # imag_4 = image[:, :256, 256:]
        # mask_4 = mask[:256, 256:]
        # oneset_img[3] = imag_4
        # oneset_mask[3] = mask_4
        
        # print(mask.shape)
        # transform = transforms.Resize((256,256))
        # mask = transform(torch.unsqueeze(mask, 0))

        return image, mask, mask_name

    def __len__(self):
        return len(self.sat_name)

def build_data(args, imageset):
    if(args.dataset == "p1"):
        if(imageset == "train"):
            return TIMITDatasetP1(os.path.join(args.p1_path, "train_50"), train_tfm_p1)
        elif(imageset == "val"):
            return TIMITDatasetP1(os.path.join(args.p1_path, "val_50"), val_tfm_p1)
        else:
            return TIMITDatasetP1(args.p1_path, val_tfm_p1)

    else:
        if(imageset == "train"):
            return TIMITDatasetP2(os.path.join(args.p2_path, "train"))
        elif(imageset == "val"):
            return TIMITDatasetP2(os.path.join(args.p2_path, "validation"))
        else:
            return TIMITDatasetP2(args.p2_path)