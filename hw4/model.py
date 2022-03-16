

import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as T
from torch.nn import CosineSimilarity
import random


def conv_block(in_channels, out_channels):
    bn = nn.BatchNorm2d(out_channels)
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        bn,
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

class ConvNet(nn.Module):
    def __init__(self, in_channels=3, hid_channels=64, out_channels=64):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(in_channels, hid_channels),
            conv_block(hid_channels, hid_channels),
            conv_block(hid_channels, hid_channels),
            conv_block(hid_channels, out_channels)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)

class Parametric_Model(nn.Module):
    def __init__(self, num, class_num) :
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(num*1600, 800),
            nn.ReLU(),
            nn.Linear(800, 100),
            nn.ReLU(),
            nn.Linear(100, class_num),
        )

    def forward(self, a, b):
        # n = a.size(0)
        # m = b.size(0)
        # d = a.size(1)
        # if d != b.size(1):
        #     raise Exception

        # a = a.unsqueeze(1).expand(n, m, d)
        # a = b.unsqueeze(0).expand(n, m, d)

        a = a.unsqueeze(1)
        b = b.expand(a.size(0), b.size(0), b.size(1))
        # print(a.shape)
        # print(b.shape)
        x = torch.cat((a, b), 1).reshape(a.size(0), -1)
        return self.linear(x)

# @torch.no_grad()
def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    # print(x.shape)

    return -torch.pow(x - y, 2).sum(2)

def sine_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    cos = CosineSimilarity(dim=2)
    
    return cos(x, y)


class Protoypical_loss(nn.Module):
    def __init__(self, n_support, n_query, device, class_num, dist_type = "Parametric"):
        super().__init__()
        self.n_support = n_support
        self.n_query = n_query
        self.device = device
        self.class_num = class_num

        if(dist_type == "Euclidean"):
            self.ds_function = euclidean_dist
        elif(dist_type == "Sine"):
            self.ds_function = sine_dist
        else:
            self.ds_function = Parametric_Model(class_num+1, class_num).to(device)

    # @torch.no_grad()
    def forward(self, x, y):

        classes = torch.unique(y)
        n_classes = len(classes)

        sup_index = torch.zeros((n_classes, self.n_support), dtype=torch.long).to(self.device)
        que_index = torch.zeros((n_classes, self.n_query), dtype=torch.long).to(self.device)
        que_label = torch.arange(n_classes, dtype=torch.long).to(self.device)
        que_label = que_label.repeat_interleave(self.n_query)
        for i, label in enumerate(classes):
            label_index = (y==label).nonzero().flatten()
            sup_index[i] = label_index[:self.n_support]
            que_index[i] = label_index[self.n_support:]
            # que_label[que_index[i]] = i

        # print(x.dtype)
        # exit()
        sup_x_reshape = x[sup_index].reshape(-1, x.shape[-1])
        sup_x = torch.zeros((n_classes, x.shape[-1]), dtype=x.dtype).to(self.device)

        for i in range(n_classes):
            sup_x[i] = torch.mean(sup_x_reshape[i*(sup_x_reshape.shape[0]//n_classes):(i+1)*(sup_x_reshape.shape[0]//n_classes)], dim=0)
        que_x = x[que_index].reshape(-1, x.shape[-1])

        dists = self.ds_function(que_x, sup_x)
        out = dists.argmax(dim=-1)

        acc = (out == que_label).float().mean()


        return F.cross_entropy(dists, que_label), acc


class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p
    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)

class AG(nn.Module):
    def __init__(self, class_num=65):
        super().__init__()

        # self.aug = torch.nn.Sequential(
        #     RandomApply(
        #         T.ColorJitter(0.8, 0.8, 0.8, 0.2),
        #         p = 0.3
        #     ),
        #     T.RandomGrayscale(p=0.2),
        #     T.RandomHorizontalFlip(),
        #     RandomApply(
        #         T.GaussianBlur((3, 3), (1.0, 2.0)),
        #         p = 0.2
        #     ),
        #     T.RandomResizedCrop((128, 128)),
        #     T.Normalize(
        #         mean=torch.tensor([0.485, 0.456, 0.406]),
        #         std=torch.tensor([0.229, 0.224, 0.225])),
        # )
    def forward(self, x):
        return self.aug(x)

class REST(nn.Module):
    def __init__(self, class_num=65):
        super().__init__()

      
    def forward(self, x):
        return x
            
class DownStreamModel(nn.Module):
    def __init__(self, class_num=65):
        super().__init__()
        self.downstream = nn.Sequential(
            nn.ReLU(),
            nn.Linear(2048, 1000),
            nn.ReLU(),
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, 100),
            nn.ReLU(),
            nn.Linear(100, 65),

        )


    # @torch.no_grad()
    def forward(self, x):
        return self.downstream(x)
 









