import os

from pandas.io.pytables import Selection

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler

from sklearn.preprocessing import LabelEncoder

import pandas as pd
import numpy as np

from PIL import Image
filenameToPILImage = lambda x: Image.open(x)




class OfficeDataset(Dataset): 
    def __init__(self, csv_path, data_dir, le=None):
        super(OfficeDataset, self).__init__()
        self.data_dir = data_dir
        self.data_df = pd.read_csv(csv_path).set_index("id")
        self.test = False
        if(le == None):
            le = LabelEncoder()
            le.fit(self.data_df.label)
        else:
            self.test = True
        self.le = le
        self.data_df.label = le.transform(self.data_df.label)

        # self.data_df.label = pd.Categorical(pd.factorize(self.data_df.label)[0])
        self.transform = transforms.Compose([
            filenameToPILImage,
            transforms.Resize((128,128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  
        ])

    def __getitem__(self, index):
        
        path = self.data_df.loc[int(index), "filename"]
        label = self.data_df.loc[int(index), "label"]
        image = self.transform(os.path.join(self.data_dir, path))
        if(self.test == True):
            return image, label, path
        else:
            return image, label

    def __len__(self):
        return len(self.data_df)


# N-way K-shot dataset
class MiniTrainDataset(Dataset): 
    def __init__(self, csv_path, data_dir):
        super(MiniTrainDataset, self).__init__()
        self.data_dir = data_dir
        self.data_df = pd.read_csv(csv_path).set_index("id")
        self.data_df.label = pd.Categorical(pd.factorize(self.data_df.label)[0])
        self.transform = transforms.Compose([
            filenameToPILImage,
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  
        ])

    def __getitem__(self, index):
        
        path = self.data_df.loc[int(index), "filename"]
        label = self.data_df.loc[int(index), "label"]
        image = self.transform(os.path.join(self.data_dir, path))
        return image, label

    def __len__(self):
        return len(self.data_df)

class PrototypicalBatchSampler(object):
    '''
    this block of code is derived from 
    orobix/Prototypical-Networks-for-Few-shot-Learning-PyTorch
    '''
    def __init__(self, data_pd, class_per_it, num_sample, iteration):
        '''
        -data_pd: pd containing data info
        -class_per_it: the number to classifiy per it, which means k shot
        -num_sample: the number to sample (support + query) in total
        -iteration: the total episodes per epoch
        '''
        super(PrototypicalBatchSampler, self).__init__()

        self.sample_per_class = num_sample
        self.classes_per_it = class_per_it
        self.iterations = iteration
        self.classes, self.counts = np.unique(data_pd["label"], return_counts=True)
        self.classes = np.arange(len(self.classes))
        self.classes = torch.LongTensor(self.classes)

        self.idxs = range(len(data_pd))
        self.indexes = np.empty((len(self.classes), max(self.counts)), dtype=int) * np.nan
        self.indexes = torch.Tensor(self.indexes)
        self.numel_per_class = torch.zeros_like(self.classes)
        for idx, label in enumerate(data_pd["label"]):
            label_idx = np.argwhere(self.classes == label).item()
            self.indexes[label_idx, np.where(np.isnan(self.indexes[label_idx]))[0][0]] = idx
            self.numel_per_class[label_idx] += 1

    def __iter__(self):
        '''
        yield a batch of indexes
        '''
        spc = self.sample_per_class
        cpi = self.classes_per_it

        for it in range(self.iterations):
            batch_size = spc * cpi
            batch = torch.LongTensor(batch_size)
            c_idxs = torch.randperm(len(self.classes))[:cpi]
            for i, c in enumerate(self.classes[c_idxs]):
                s = slice(i * spc, (i + 1) * spc)
                # FIXME when torch.argwhere will exists
                label_idx = torch.arange(len(self.classes)).long()[self.classes == c].item()
                sample_idxs = torch.randperm(self.numel_per_class[label_idx])[:spc]
                batch[s] = self.indexes[label_idx][sample_idxs]
            batch = batch[torch.randperm(len(batch))]

            yield batch

    def __len__(self):
        '''
        returns the number of iterations (episodes) per epoch
        '''
        return self.iterations

