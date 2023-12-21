import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import cv2

class CustomImageDataset(Dataset):
    def __init__(self, vintage_dir, color_dir, transform=None):
        self.vintage_dir = vintage_dir
        self.color_dir = color_dir
        self.transform = transform
        self.image_filenames = [f for f in os.listdir(vintage_dir) if os.path.isfile(os.path.join(vintage_dir, f))]
        self.image_filenames.sort()

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        vintage_img_name = os.path.join(self.vintage_dir, self.image_filenames[idx])
        color_img_name = os.path.join(self.color_dir, self.image_filenames[idx])

        vintage_image = Image.open(vintage_img_name).convert('RGB')
        color_image = Image.open(color_img_name).convert('RGB')

        # Convert PIL Images to NumPy arrays
        vintage_image_np = np.array(vintage_image)
        color_image_np = np.array(color_image)

        # Convert color image to Lab color space
        color_image_lab = cv2.cvtColor(color_image_np, cv2.COLOR_RGB2LAB)
        L = color_image_lab[:, :, 0]
        ab = color_image_lab[:, :, 1:]

        # Apply transformations
        if self.transform:
            vintage_image = self.transform(vintage_image_np)
            L = self.transform(L).unsqueeze(0)
            ab = self.transform(ab)

        return {'L': L, 'ab': ab, 'vintage': vintage_image}

class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x

class GeneratorUNet(nn.Module):
    def __init__(self):
        super(GeneratorUNet, self).__init__()

        # Downsampling
        self.down1 = UNetDown(1, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        # Upsampling
        self.up1 = UNetUp(512, 256, dropout=0.5)
        self.up2 = UNetUp(256 * 2, 128, dropout=0.5)
        self.up3 = UNetUp(128 * 2, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(64 * 2, 2, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        
        u1 = self.up1(d4, d3)
        u2 = self.up2(u1, d2)
        u3 = self.up3(u2, d1)
        
        return self.final(u3)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(4, 64, normalization=False),  # 4-channel input for concatenated Lab image
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)  # Output is a matrix of scores
        )

    def forward(self, img):
        return self.model(img)

