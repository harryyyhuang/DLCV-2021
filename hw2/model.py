import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F

from torch.autograd import Function

# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         m.weight.data.normal_(0.0, 0.02)
#     elif classname.find('BatchNorm') != -1:
#         m.weight.data.normal_(1.0, 0.02)
#         m.bias.data.fill_(0)

class Generator(nn.Module):
    def __init__(self, in_dim, dim=64):
        super(Generator, self).__init__()
        def dconv_bn_relu(in_dim, out_dim, layer_shape):
            return nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, 4, 2,
                                   padding=1, bias=False),
                # nn.BatchNorm2d(out_dim),
                nn.LayerNorm((out_dim, layer_shape, layer_shape)),
                nn.ReLU()
            )
        def dd_bn_relu(in_dim, out_dim, layer_shape):
            return nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, 4, 1,
                                   padding=0, bias=False),
                # nn.BatchNorm2d(out_dim),
                nn.LayerNorm((out_dim, layer_shape, layer_shape)),
                nn.ReLU()
            )
        self.l1 = nn.Sequential(
            nn.Linear(in_dim, dim * 8 * 4 * 4, bias=False),
            # nn.BatchNorm1d(dim * 8 * 4 * 4),
            nn.LayerNorm(dim * 8 * 4 * 4),
            nn.ReLU()
        )

        self.l2_5 = nn.Sequential(
            # dconv_bn_relu(in_dim, dim * 4, 2),
            dd_bn_relu(in_dim, dim * 8, 4),
            # dconv_bn_relu(dim * 4, dim * 8, 4),
            dconv_bn_relu(dim * 8, dim * 4, 8),
            dconv_bn_relu(dim * 4, dim * 2, 16),
            dconv_bn_relu(dim * 2, dim, 32),
            nn.ConvTranspose2d(dim, 3, 4, 2, padding=1, bias=False),
            nn.Tanh()
        )
        # self.dd_1 = dd_bn_relu(in_dim, dim * 4, 2)
        # self.dd_2 = dd_bn_relu(dim * 4, dim * 8, 4)

        # self.apply(weights_init)


        

    def forward(self, x):
        # y = self.l1(x)
        # y = x.view(x.size(0), -1, 64, 64)
        # y = self.dd_1(x)
        # y = self.dd_2(y)
        # y = x.view(x.size(0), -1, 4, 4)
        y = self.l2_5(x)
        return y

class Discriminator(nn.Module):
    def __init__(self, in_dim, dim=64):
        super(Discriminator, self).__init__()

        def conv_bn_lrelu(in_dim, out_dim, layer_shape):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 4, 2, 1),
                # nn.BatchNorm2d(out_dim),
                nn.LayerNorm((out_dim, layer_shape, layer_shape)),
                nn.LeakyReLU(0.2)
            )

        self.ls = nn.Sequential(
            nn.Conv2d(in_dim, dim, 4, 2, 1), 
            nn.LeakyReLU(0.2),
            conv_bn_lrelu(dim, dim * 2, 16),
            conv_bn_lrelu(dim * 2, dim * 4, 8),
            conv_bn_lrelu(dim * 4, dim * 8, 4),
            nn.Conv2d(dim * 8, 1, 4),
        
        )
        # self.apply(weights_init)


        
    def forward(self, x):
        y = self.ls(x)
        y = y.view(-1)
        return y




class ACGenerator(nn.Module):
    def __init__(self, in_dim, dim=64):
        super(ACGenerator, self).__init__()
        def dconv_bn_relu(in_dim, out_dim, layer_shape):
            return nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, 4, 2,
                                   padding=1, bias=False),
                # nn.BatchNorm2d(out_dim),
                nn.LayerNorm((out_dim, layer_shape, layer_shape)),
                nn.ReLU()
            )
        def dd_bn_relu(in_dim, out_dim, layer_shape):
            return nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, 4, 1,
                                   padding=0, bias=False),
                # nn.BatchNorm2d(out_dim),
                nn.LayerNorm((out_dim, layer_shape, layer_shape)),
                nn.ReLU()
            )
        self.l1 = nn.Sequential(
            nn.Linear(in_dim, dim * 8 * 7 * 7, bias=False),
            # nn.BatchNorm1d(dim * 8 * 7 * 7),
            nn.LayerNorm(dim * 8 * 7 * 7),
            nn.ReLU()
        )

        self.l2_5 = nn.Sequential(
            # dd_bn_relu(in_dim, dim * 8, 4),
            dconv_bn_relu(dim * 8, dim * 4, 14),
            # dconv_bn_relu(dim * 4, dim * 2, 16),
            # dconv_bn_relu(dim * 2, dim, 32),
            nn.ConvTranspose2d(dim*4, 3, 4, 2, padding=1, bias=False),
            nn.Tanh()
        )


    def onehorEncode(self, label):
        hot = torch.zeros((label.size(0), 10)).cuda()
        hot.scatter_(1, label.type(torch.long).view(-1, 1), 1)
        return hot

    def forward(self, x, label):
        y = self.onehorEncode(label)
        y = torch.cat((x, y), axis=1)
        y = self.l1(y)
        y = y.view(y.size(0), -1, 7, 7)
        y = self.l2_5(y)
        return y

class ACDiscriminator(nn.Module):
    def __init__(self, in_dim, dim=64):
        super(ACDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

        # self.ls = nn.Sequential(
        #     nn.Conv2d(3, 6, 5),
        #     nn.MaxPool2d(2, 2),
        #     nn.Conv2d(6, 16, 5),
        #     nn.Linear(16 * 4 * 4, 128),
        #     nn.Linear(128, 64),
        # )

        self.ls_label = nn.Sequential(
            nn.Linear(64, 10)
        )

        self.ls_discrem = nn.Sequential(
            nn.Linear(64, 1)
        )

        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        y_label = self.ls_label(x)
        y_dis = self.ls_discrem(x)
        y_dis = y_dis.view(-1)
        return y_label, y_dis

class DANNFeatur(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.conv3 = nn.Conv2d(16, 32, 3)

        self.convs = nn.Sequential(
            nn.Conv2d(3, 64, 5),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(64, 50, 5),
            nn.Dropout2d(),
            nn.BatchNorm2d(50),
            nn.MaxPool2d(2),
            nn.ReLU(),
            

        )


    def forward(self, x):
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = self.pool(F.relu(self.conv3(x)))
        x = self.convs(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        # print(x.shape)
        return x
class DANNClass(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(50*4*4 , 100),
        
        self.fc2 = nn.Linear(256 , 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

        self.classifier = nn.Sequential(
            nn.Linear(50 * 4 * 4, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Dropout2d(),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, 10),
        )

    def forward(self, x):
        return self.classifier(x)

class DANNDomain(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32 , 256)
        self.fc2 = nn.Linear(256 , 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 2)
        self.domainer = nn.Sequential(
            nn.Linear(50 * 4 * 4, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, 2),
        )

    def forward(self, x):
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = self.fc4(x)
        return self.domainer(x)




class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class DANNModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.FeatureExtract = DANNFeatur()
        self.classifier = DANNClass()
        self.domainClassifier = DANNDomain()

    def forward(self, x, alpha=1.0):
        features = self.FeatureExtract(x)
        class_label = self.classifier(features)
        reverse_feature = ReverseLayerF.apply(features, alpha)
        domain_label = self.domainClassifier(reverse_feature)
        # return features
        return class_label, domain_label


def build_model(args):
    if(args.model == "p1"):
        G = Generator(in_dim=128).cuda()
        D = Discriminator(3).cuda()
        D.load_state_dict(torch.load("./D_strong.pth"))
        G.load_state_dict(torch.load("./G_strong.pth"))
        # for param_name, pt_param in pretrain_VAE.items():
        #     if(param_name[2:] in G.state_dict()):
        #         G.state_dict()[param_name[2:]].copy_(pt_param)

        # for param_name, pt_param in pretrain_VAE.items():
        #     if(param_name[2:] in D.state_dict()):
        #         D.state_dict()[param_name[2:]].copy_(pt_param)

        return G, D

    elif(args.model == "p2"):
        G = ACGenerator(in_dim=110).cuda()
        D = ACDiscriminator(3).cuda()
        D.load_state_dict(torch.load("./D_p2.pth"))
        G.load_state_dict(torch.load("./G_p2.pth"))
        # for param_name, pt_param in pretrain_VAE.items():
        #     if(param_name[2:] in G.state_dict()):
        #         G.state_dict()[param_name[2:]].copy_(pt_param)

        # for param_name, pt_param in pretrain_VAE.items():
        #     if(param_name[2:] in D.state_dict()):
        #         D.state_dict()[param_name[2:]].copy_(pt_param)

        return G, D
    else:
        DANN = DANNModel()
        if(args.data_name == "mnistm"):
            DANN.load_state_dict(torch.load("./DANN3-1.pth"))
        elif(args.data_name == "usps"):
            DANN.load_state_dict(torch.load("./DANN3-2.pth"))
        else:
            DANN.load_state_dict(torch.load("./DANN3-3.pth"))
        return DANN
