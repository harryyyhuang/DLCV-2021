import torch
import torch.nn as nn
import torchvision.models as models
from mean_iou_evaluate import read_masks
import torch.nn.functional as F


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.myModel = models.vgg16_bn(pretrained=True)

        self.myModel.classifier[6] = nn.Linear(4096, 50)
        self.linear = nn.Sequential(
            nn.ReLU(),
            nn.Linear(2048, 50),
        )

    def forward(self, x):
        x = self.myModel(x)
        return x


class Segmentator(nn.Module):
    def __init__(self):
        super(Segmentator, self).__init__()
        backbone = models.vgg16(pretrained=True)
        self.myModel = nn.Sequential(*(list(backbone.children())[:-2]))

        self.conv = nn.Sequential(
            nn.Conv2d(512, 4096, 1, 1),
            nn.Conv2d(4096, 4096, 1, 1),
            nn.Conv2d(4096, 7, 1, 1),
        )

        self.up_conv = nn.ConvTranspose2d(7, 7, 64, 32, 16, bias=False)
    
    def forward(self, x):
        x = self.myModel(x)
        x = self.conv(x)
        x = self.up_conv(x)
        return x


class RestNetFCN(nn.Module):
    def __init__(self):
        super(RestNetFCN, self).__init__()
        backbone = models.segmentation.fcn_resnet101(pretrained=True, num_classes=7)
        
    def forward(self, x):
        x = self.myModel(x)
        x = self.FCNHead(x['out'])
       
        return x

class dec_Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(dec_Block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 1),
            nn.ReLU(),
            # nn.BatchNorm2d(out_channel),
            #nn.Conv2d(out_channel, out_channel, 3, 1, 1),
            #nn.ReLU(),
            #nn.BatchNorm2d(out_channel),
        )
    def forward(self, x):
        return self.block(x)


class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        backbone = models.vgg16(pretrained = True)
        pretrain_list = list(backbone.children())[:-2]

        # print(pretrain_list)
        self.myEncoder = nn.ModuleList(*(pretrain_list))
        # print(self)
        # for layer in self.myEncoder:
        #     print(layer)
        #print(self.myEncoder)
        #exit()
        self.lastLayer = nn.Sequential(
            nn.Conv2d(512, 512, 1, 1),
            nn.ReLU(),
            # nn.BatchNorm2d(512),
        # )
#            nn.Conv2d(1024, 512, 1, 1),
 #           nn.ReLU(),
  #          nn.BatchNorm2d(512),
        )

        channels = [64, 128, 256, 512, 512, 512]
        channels.reverse()
        dec_channels_in = [1024, 1024, 512, 256, 128]
        dec_channels_out = [512, 512, 256, 128, 64]

        self.upConv = nn.ModuleList([nn.ConvTranspose2d(channels[i], channels[i+1], 2, 2) 
                                         for i in range(len(channels)-1)])
        #self.upConv = nn.ModuleList([nn.Upsample(channels[i]) 
        #                                for i in range(len(channels))])
        self.dec_Conv = nn.ModuleList([dec_Block(dec_channels_in[i], dec_channels_out[i])
                                        for i in range(len(dec_channels_in))])

        self.outLayer = nn.Sequential(
            nn.Conv2d(64, 7, 1),
            #nn.ReLU(),
            #nn.BatchNorm2d(7),
            #nn.Upsample(512)
        )

    def forward(self, x):
        encoder_features = []
        for encoder in self.myEncoder:
            if(isinstance(encoder, nn.MaxPool2d)):
                encoder_features.append(x)
            x = encoder(x)
        encoder_features.reverse()
        x = self.lastLayer(x)

        for decoder_up, decoder_conv, encoder_filt in zip(self.upConv, self.dec_Conv, encoder_features):
            x = decoder_up(x)
            #print(x.shape)
            #print(encoder_filt.shape)
            #print(x.shape)
            x = torch.cat([x, encoder_filt], dim=1)
            #print(x.shape)
            x = decoder_conv(x)

        x = self.outLayer(x)
        # x = nn.Upsample
        x = nn.functional.interpolate(x, (512, 512))
        return x


    


def build_model(args):
    if(args.dataset == "p1"):
        return Classifier()
    else:
        return Unet()
