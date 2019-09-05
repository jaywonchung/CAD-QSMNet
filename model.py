import torch
import torch.nn as nn

from constants import *


class DoubleConv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 5, stride=1, padding=2),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 5, stride=1, padding=2),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                m.bias.data.fill_(0.)

    def forward(self, x):
        x = self.conv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.conv1 = DoubleConv(in_ch, in_ch)
        self.conv2 = nn.Conv3d(in_ch, out_ch, 5, stride=1, padding=2)
        self.pool = nn.MaxPool3d(2)

    def forward(self, x):
        x_conv = self.conv1(x)
        x_pool = self.conv2(x_conv)
        x_pool = self.pool(x_pool)
        return x_conv, x_pool


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Up, self).__init__()
        self.upconv = nn.ConvTranspose3d(in_ch, in_ch//2, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(m.weight)
                m.bias.data.fill_(0.)

    def forward(self, x, x_conv, x_pool):
        x = self.upconv(x)
        x = torch.cat([x, x_conv], dim=1)
        x = self.conv(x)
        x = x - x_pool
        return x


class Middle(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Middle, self).__init__()
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = x - self.conv(x)
        return x


class In(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(In, self).__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, 5, stride=1, padding=2)

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(m.weight)
                m.bias.data.fill_(0.)

    def forward(self, x):
        x = self.conv(x)
        return x


class Out(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Out, self).__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, 1)

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(m.weight)
                m.bias.data.fill_(0.)

    def forward(self, x):
        x = self.conv(x)
        return x


class QSMNet(nn.Module):
    def __init__(self):
        super(QSMNet, self).__init__()
        self.inconv = In(1, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        self.middle = Middle(512, 512)
        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)
        self.up4 = Up(64, 32)
        self.outconv = Out(32, 1)

    def forward(self, x):
        skip1 = x
        cat1, skip2 = self.down1(self.inconv(x))
        cat2, skip3 = self.down2(skip2)
        cat3, skip4 = self.down3(skip3)
        cat4, skip5 = self.down4(skip4)

        out = self.middle(skip5)

        out = self.up1(out, cat4, skip4)
        out = self.up2(out, cat3, skip3)
        out = self.up3(out, cat2, skip2)
        out = self.up4(out, cat1, skip1)
        out = self.outconv(out)

        return out
