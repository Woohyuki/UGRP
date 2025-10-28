"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.vgg16_bn import vgg16_bn, init_weights
from model import Layers

class CRAFT(nn.Module):
    def __init__(self, amp=False):
        super(CRAFT, self).__init__()

        self.amp = amp

        """ Base network """
        # self.basenet = vgg16_bn(pretrained, freeze)
        self.basenet = vgg16_bn()


        """ U network """
        self.upconv1 = torch.nn.Sequential(
            Layers.QMobilenet2d(
                in_ch=128+128,
                out_ch=128,
                kernel_size=3,
                stride=1,
                min_val=0, max_val=1e9,
                act1=nn.Identity()
            ),
            Layers.QMobilenet2d(
                in_ch=128,
                out_ch=48,
                kernel_size=3,
                stride=1,
                min_val=0, max_val=1e9,
                act1=nn.Identity()
            ),
        )
        self.upconv2 = torch.nn.Sequential(
            Layers.QMobilenet2d(
                in_ch=64+48,
                out_ch=64,
                kernel_size=3,
                stride=1,
                min_val=0, max_val=1e9,
                act1=nn.Identity()
            ),
            Layers.QMobilenet2d(
                in_ch=64,
                out_ch=24,
                kernel_size=3,
                stride=1,
                min_val=0, max_val=1e9,
                act1=nn.Identity()
            ),
        )
        self.upconv3 = torch.nn.Sequential(
            Layers.QMobilenet2d(
                in_ch=32+24,
                out_ch=32,
                kernel_size=3,
                stride=1,
                min_val=0, max_val=1e9,
                act1=nn.Identity()
            ),
            Layers.QMobilenet2d(
                in_ch=32,
                out_ch=12,
                kernel_size=3,
                stride=1,
                min_val=0, max_val=1e9,
                act1=nn.Identity()
            ),
        )
        self.upconv4 = torch.nn.Sequential(
            Layers.QMobilenet2d(
                in_ch=16+12,
                out_ch=16,
                kernel_size=3,
                stride=1,
                min_val=0, max_val=1e9,
                act1=nn.Identity()
            ),
            Layers.QMobilenet2d(
                in_ch=16,
                out_ch=8,
                kernel_size=3,
                stride=1,
                min_val=0, max_val=1e9,
                act1=nn.Identity()
            ),
        )

        self.conv_cls = nn.Sequential(

            Layers.QMobilenet2d(
                in_ch=8,
                out_ch=16,
                kernel_size=3,
                stride=1,
                min_val=0, max_val=1e9,
                BNfold=False,
                act1=nn.Identity()
            ),
            Layers.QMobilenet2d(
                in_ch=16,
                out_ch=16,
                kernel_size=3,
                stride=1,
                min_val=0, max_val=1e9,
                BNfold=False,
                act1=nn.Identity()
            ),
            Layers.QMobilenet2d(
                in_ch=16,
                out_ch=8,
                kernel_size=3,
                stride=1,
                min_val=0, max_val=1e9,
                BNfold=True,
                act1=nn.Identity()
            ),
            Layers.QPointwiseConv2d(8,8),
            nn.ReLU(inplace=True),
            Layers.QPointwiseConv2d(8,2),
            Layers.QSTEHardSigmoid(slope = 1/32)
        )

        init_weights(self.basenet.modules())
        init_weights(self.upconv1.modules())
        init_weights(self.upconv2.modules())
        init_weights(self.upconv3.modules())
        init_weights(self.upconv4.modules())
        init_weights(self.conv_cls.modules())
        
    def forward(self, x):
        """ Base network """
        if self.amp:
            with torch.cuda.amp.autocast():
                sources = self.basenet(x)

                y = torch.cat([sources[0], sources[1]], dim=1)
                y = self.upconv1(y)

                y = F.interpolate(y, size=sources[2].size()[2:], mode='bilinear', align_corners=False)
                y = torch.cat([y, sources[2]], dim=1)
                y = self.upconv2(y)

                y = F.interpolate(y, size=sources[3].size()[2:], mode='bilinear', align_corners=False)
                y = torch.cat([y, sources[3]], dim=1)
                y = self.upconv3(y)

                y = F.interpolate(y, size=sources[4].size()[2:], mode='bilinear', align_corners=False)
                y = torch.cat([y, sources[4]], dim=1)
                feature = self.upconv4(y)

                y = self.conv_cls(feature)

                return y.permute(0,2,3,1)
                
        else:

            sources = self.basenet(x)

            y = torch.cat([sources[0], sources[1]], dim=1)
            y = self.upconv1(y)

            y = F.interpolate(y, size=sources[2].size()[2:], mode='bilinear', align_corners=False)
            y = torch.cat([y, sources[2]], dim=1)
            y = self.upconv2(y)

            y = F.interpolate(y, size=sources[3].size()[2:], mode='bilinear', align_corners=False)
            y = torch.cat([y, sources[3]], dim=1)
            y = self.upconv3(y)

            y = F.interpolate(y, size=sources[4].size()[2:], mode='bilinear', align_corners=False)
            y = torch.cat([y, sources[4]], dim=1)
            feature = self.upconv4(y)

            y = self.conv_cls(feature)

            return y.permute(0, 2, 3, 1)

if __name__ == '__main__':
    model = CRAFT(pretrained=True).cuda()
    output, _ = model(torch.randn(1, 3, 768, 768).cuda())
    print(output.shape)