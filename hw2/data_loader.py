from torch.utils.data import Dataset
import os
from PIL import Image
import torchvision.transforms as transforms
import torchvision
import numpy as np


train_tfm_p1 = transforms.Compose([
    # transforms.Resize((64, 64)),
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

val_tfm_p1 = transforms.Compose([
    # transforms.Resize((64, 64)),
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


train_tfm_p2 = transforms.Compose([
    # transforms.Resize((64, 64)),
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

test_tfm_p2 = transforms.Compose([
    # transforms.Resize((64, 64)),
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


class FaceDataset(Dataset):
    def __init__(self, data_dir, transform):
        super(FaceDataset, self).__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.file_names = sorted(os.listdir(data_dir))
        self.num_names = len(self.file_names)

    def __getitem__(self, idx):
        name = self.file_names[idx]
        img = torchvision.io.read_image(os.path.join(self.data_dir, name))
        img = self.transform(img)
        return img
    
    def __len__(self):
        return self.num_names

class minstm(Dataset):
    def __init__(self, data_dir, csv_path, transform):
        super(minstm, self).__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.file_names = sorted(os.listdir(data_dir))
        self.label = np.genfromtxt(csv_path, delimiter=',', skip_header=1)

    def __getitem__(self, idx):
        name = self.file_names[idx]
        img = torchvision.io.read_image(os.path.join(self.data_dir, name), mode=torchvision.io.image.ImageReadMode.RGB)
        img = self.transform(img)
        return (img, self.label[idx][1], name)
    def __len__(self):
        return len(self.file_names[:2000])

class minstm_test(Dataset):
    def __init__(self, data_dir, transform):
        super(minstm_test, self).__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.file_names = sorted(os.listdir(data_dir))

    def __getitem__(self, idx):
        name = self.file_names[idx]
        img = torchvision.io.read_image(os.path.join(self.data_dir, name), mode=torchvision.io.image.ImageReadMode.RGB)
        img = self.transform(img)
        return (img, name)
    def __len__(self):
        return len(self.file_names)


def build_data(args, imageset):
    if(args.dataset == "p1"):
        if(imageset == "train"):
            return FaceDataset(os.path.join(args.p1_path, "train"), train_tfm_p1)
        elif(imageset == "gen"):
            return FaceDataset(args.p1_gen_path, val_tfm_p1)
        elif(imageset == "test"):
            return FaceDataset(os.path.join(args.p1_path, "test"), val_tfm_p1)
        else:
            return FaceDataset(args.p1_path, val_tfm_p1)

    elif(args.dataset == "p2"):
        if(imageset == "train"):
            csv_path = os.path.join(args.p2_path, "train.csv")
            image_path = os.path.join(args.p2_path, "train")
            return minstm(image_path, csv_path, train_tfm_p2)
        elif(imageset == "test"):
            csv_path = os.path.join(args.p2_path, "test.csv")
            image_path = os.path.join(args.p2_path, "test")
            return minstm(image_path, csv_path, test_tfm_p2)

    elif(args.dataset == "p3"):
        if(imageset == "mnistm2usps"):
            csv_path_train = os.path.join(os.path.join(args.p2_path, "mnistm"), "train.csv")
            csv_path_test = os.path.join(os.path.join(args.p2_path, "usps"), "train.csv")
            image_path_mnis = os.path.join(os.path.join(args.p2_path, "mnistm"), "train")
            image_path_usps = os.path.join(os.path.join(args.p2_path, "usps"), "train")
            return minstm(image_path_mnis, csv_path_train, train_tfm_p2), minstm(image_path_usps, csv_path_test, train_tfm_p2)
        elif(imageset == "usps"):
            image_path_usps = os.path.join(os.path.join(args.p2_path, "usps"), "test")
            csv_path_test = os.path.join(os.path.join(args.p2_path, "usps"), "test.csv")
            return minstm(image_path_usps, csv_path_test, train_tfm_p2)
        elif(imageset == "svhn2mnist"):
            csv_path_train = os.path.join(os.path.join(args.p2_path, "svhn"), "train.csv")
            csv_path_test = os.path.join(os.path.join(args.p2_path, "mnistm"), "train.csv")
            image_path_mnis = os.path.join(os.path.join(args.p2_path, "svhn"), "train")
            image_path_usps = os.path.join(os.path.join(args.p2_path, "mnistm"), "train")
            return minstm(image_path_mnis, csv_path_train, train_tfm_p2), minstm(image_path_usps, csv_path_test, train_tfm_p2)
        elif(imageset == "mnist"):
            image_path_usps = os.path.join(os.path.join(args.p2_path, "mnistm"), "test")
            csv_path_test = os.path.join(os.path.join(args.p2_path, "mnistm"), "test.csv")
            return minstm(image_path_usps, csv_path_test, train_tfm_p2)
        elif(imageset == "usps2svhn"):
            csv_path_train = os.path.join(os.path.join(args.p2_path, "usps"), "train.csv")
            csv_path_test = os.path.join(os.path.join(args.p2_path, "svhn"), "train.csv")
            image_path_mnis = os.path.join(os.path.join(args.p2_path, "usps"), "train")
            image_path_usps = os.path.join(os.path.join(args.p2_path, "svhn"), "train")
            return minstm(image_path_mnis, csv_path_train, train_tfm_p2), minstm(image_path_usps, csv_path_test, train_tfm_p2)
        elif(imageset == "svhn"):
            image_path_usps = os.path.join(os.path.join(args.p2_path, "svhn"), "test")
            csv_path_test = os.path.join(os.path.join(args.p2_path, "svhn"), "test.csv")
            return minstm(image_path_usps, csv_path_test, train_tfm_p2)
        
    else:
        return minstm_test(args.data_path, train_tfm_p2)
