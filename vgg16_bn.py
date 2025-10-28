import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision
from torchvision import models
from packaging import version
from model import Layers

def init_weights(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
            #init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

class vgg16_bn(torch.nn.Module):
    def __init__(self):

        super(vgg16_bn, self).__init__()

        self.slice1 = torch.nn.Sequential() #16
        self.slice2 = torch.nn.Sequential() #24
        self.slice3 = torch.nn.Sequential() #32
        self.slice4 = torch.nn.Sequential() #64
        #self.slice5 = torch.nn.Sequential()

        self.slice1 = torch.nn.Sequential(

            Layers.QConv2d(
                in_ch=1,
                out_ch=16,
                kernel_size=3,
                stride=1,
                pad_size=1,
            ),
            Layers.QBatchNorm2d(16),

            Layers.QMobilenet2d(
                in_ch=16,
                out_ch=16,
                kernel_size=3,
                stride=1,
                min_val=0, max_val=1e9,
                act1=nn.Identity()
            ),
            nn.MaxPool2d(kernel_size=2),

            Layers.QInvertedResidualBlock(
                in_ch=16,
                out_ch=32,
                expand_ratio=2,
                stride=1,
                kernel_size=3,
                pad_size=1,
                min_val=0, max_val=1e9,
                act2 = nn.Identity(),
                residual = False
            ),
            nn.ReLU(inplace=True),

            Layers.QMobilenet2d(
                in_ch=32,
                out_ch=16,
                kernel_size=3,
                stride=1,
                min_val=0, max_val=1e9,
                act1=nn.Identity(), act2=nn.Identity()
            ),
        )

        self.slice2 = torch.nn.Sequential(
            
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            Layers.QInvertedResidualBlock(
                in_ch=16,
                out_ch=64,
                expand_ratio=4,
                stride=1,
                kernel_size=3,
                pad_size=1,
                min_val=0, max_val=1e9,
                act2 = nn.Identity(),
                residual = False
            ),
            nn.ReLU(inplace=True),

            Layers.QMobilenet2d(
                in_ch=64,
                out_ch=32,
                kernel_size=3,
                stride=1,
                min_val=0, max_val=1e9,
                act1=nn.Identity(), act2=nn.Identity()
            ),
        )

        self.slice3 = torch.nn.Sequential(

            nn.ReLU(inplace=True),
            Layers.QMobilenet2d(
                in_ch=32,
                out_ch=32,
                kernel_size=3,
                stride=1,
                min_val=0, max_val=1e9,
                act1=nn.Identity()
            ),
            nn.MaxPool2d(kernel_size=2),
            
            Layers.QInvertedResidualBlock(
                in_ch=32,
                out_ch=128,
                expand_ratio=4,
                stride=1,
                kernel_size=3,
                pad_size=1,
                min_val=0, max_val=1e9,
                act2 = nn.Identity(),
                residual = False
            ),
            nn.ReLU(inplace=True),

            Layers.QMobilenet2d(
                in_ch=128,
                out_ch=64,
                kernel_size=3,
                stride=1,
                min_val=0, max_val=1e9,
                act1=nn.Identity(), act2=nn.Identity()
            ),
        )

        self.slice4 = torch.nn.Sequential(

            nn.ReLU(inplace=True),
            Layers.QMobilenet2d(
                in_ch=64,
                out_ch=64,
                kernel_size=3,
                stride=1,
                min_val=0, max_val=1e9,
                act1=nn.Identity()
            ),
            nn.MaxPool2d(kernel_size=2),
            
            Layers.QInvertedResidualBlock(
                in_ch=64,
                out_ch=256,
                expand_ratio=4,
                stride=1,
                kernel_size=3,
                pad_size=1,
                min_val=0, max_val=1e9,
                act2 = nn.Identity(),
                residual = False
            ),
            nn.ReLU(inplace=True),

            Layers.QMobilenet2d(
                in_ch=256,
                out_ch=128,
                kernel_size=3,
                stride=1,
                min_val=0, max_val=1e9,
                act1=nn.Identity(), act2=nn.Identity()
            ),
        )
        
        self.slice5 = torch.nn.Sequential(
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),

            Layers.QPointwiseConv2d(128,256),
            Layers.QBatchNorm2d(256),
            nn.ReLU(inplace=True),

            Layers.QDepthwiseConv2d(256, kernel_size=2, dilation=2, stride=1, pad_size=3),
            Layers.QBatchNorm2d(256),
            Layers.QDepthwiseConv2d(256, kernel_size=2, dilation=2, stride=1, pad_size=0),
            Layers.QBatchNorm2d(256),
            Layers.QDepthwiseConv2d(256, kernel_size=2, dilation=2, stride=1, pad_size=0),
            Layers.QBatchNorm2d(256),

            Layers.QPointwiseConv2d(256,256),
            Layers.QBatchNorm2d(256),
            nn.ReLU(inplace=True),

            Layers.QMobilenet2d(
                in_ch=256,
                out_ch=128,
                kernel_size=3,
                stride=1,
                min_val=0, max_val=1e9,
                act1=nn.Identity(), act2=nn.Identity()
            ),
        )

        init_weights(self.slice1.modules())
        init_weights(self.slice2.modules())
        init_weights(self.slice3.modules())
        init_weights(self.slice4.modules())
        init_weights(self.slice5.modules())

    def forward(self, X):

        h = self.slice1(X)
        h_relu2_2 = h

        h = self.slice2(h)
        h_relu3_2 = h

        h = self.slice3(h)
        h_relu4_3 = h

        h = self.slice4(h)
        h_relu5_3 = h

        h = self.slice5(h)
        h_fc7 = h
        
        return h_fc7, h_relu5_3, h_relu4_3, h_relu3_2, h_relu2_2
