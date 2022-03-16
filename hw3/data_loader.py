from  torch.utils.data import Dataset
import os
import PIL
from PIL import Image
import torchvision.transforms as transforms


train_tfm_p1 = transforms.Compose([
    transforms.RandomResizedCrop(384),
    transforms.RandomAffine(30),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(0.5, 0.5, 0.5, 0.5),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
])

val_tfm_p1 = transforms.Compose([
    transforms.Resize(384, interpolation=PIL.Image.BICUBIC),
    transforms.CenterCrop(384),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
])


class VITDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.files_name = os.listdir(data_dir)
        self.transform = transform

    def __getitem__(self, idx):
        data_name = self.files_name[idx]
        image = Image.open(os.path.join(self.data_dir, data_name)).convert("RGB")
        split_name = data_name.split("_")
        label = int(split_name[0])

        if self.transform is not None:
            image = self.transform(image)
        # print(image.shape)
        # print(data_name)
        return image, label, data_name

    def __len__(self):
        return len(self.files_name)



def build_data(args, imageset):
    if(args.dataset == "p1"):
        if(imageset == "train"):
            return VITDataset(os.path.join(args.p1_path, "train"), train_tfm_p1)
        elif(imageset == "val"):
            return VITDataset(os.path.join(args.p1_path, "val"), val_tfm_p1)
        else:
            return VITDataset(args.p1_path, val_tfm_p1)