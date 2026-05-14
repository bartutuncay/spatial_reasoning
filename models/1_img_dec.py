import torch
import torch.nn as nn
import torch.nn.functional as F

class UpBlock(nn.Module):
    def __init__(self,in_ch,out_ch):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))
        
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.block(x)


class ImgDec(nn.Module):
    def __init__(self,latent_dim,out_channels,base_channels,out_hw=(192,256),expansion=4):
        super().__init__()
        H, W = out_hw

        self.h0 = H // 32
        self.w0 = W // 32
        self.c0 = 512 * expansion

        self.fc = nn.Linear(latent_dim, self.c0 * self.h0 * self.w0)

        self.up1 = UpBlock(self.c0, 256)
        self.up2 = UpBlock(256, 128)
        self.up3 = UpBlock(128, 64)
        self.up4 = UpBlock(64, 32)
        self.up5 = UpBlock(32, 16)

        self.to_rgb = nn.Conv2d(16, out_channels, kernel_size=3, padding=1)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(z.size(0), self.c0, self.h0, self.w0)   # e.g. (B, 2048, 7, 7)

        x = self.up1(x)   # 7 -> 14
        x = self.up2(x)   # 14 -> 28
        x = self.up3(x)   # 28 -> 56
        x = self.up4(x)   # 56 -> 112
        x = self.up5(x)   # 112 -> 224

        x = self.to_rgb(x)
        return torch.sigmoid(x)
