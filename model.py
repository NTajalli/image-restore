import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from skimage.color import rgb2lab


class CustomImageDataset(Dataset):
    def __init__(self, vintage_dir, color_dir, transform=None, is_train=True):
        self.vintage_dir = vintage_dir
        self.color_dir = color_dir
        self.is_train = is_train
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

        # Apply transformations
        if self.transform:
            if self.is_train:
                # Apply the same random transformation to both vintage and color images
                seed = np.random.randint(2147483647)
                np.random.seed(seed)
                torch.manual_seed(seed)
                vintage_image = self.transform(vintage_image)

                np.random.seed(seed)
                torch.manual_seed(seed)
                color_image = self.transform(color_image)
            else:
                vintage_image = self.transform(vintage_image)
                color_image = self.transform(color_image)

        # Convert color image to Lab color space
        color_image_np = np.array(color_image)
        
        # Permute dimensions from (C, H, W) to (H, W, C) for skimage compatibility
        color_image_np = np.transpose(color_image_np, (1, 2, 0))

        color_image_lab = rgb2lab(color_image_np).astype("float32")  # Convert RGB to Lab
        L = color_image_lab[:, :, 0] / 50. - 1.  # Normalize L channel to [-1, 1]
        ab = color_image_lab[:, :, 1:] / 110.  # Normalize ab channels to [-1, 1]

        # Convert to tensors
        L = torch.from_numpy(L).unsqueeze(0).float()
        ab = torch.from_numpy(ab).permute(2, 0, 1).float()

        return {'L': L, 'ab': ab, 'vintage': vintage_image}

class UnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, submodule=None, input_channels=None, use_dropout=False, innermost=False, outermost=False):
        super(UnetBlock, self).__init__()
        self.outermost = outermost
        if input_channels is None: 
            input_channels = in_channels

        downconv = nn.Conv2d(input_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = nn.BatchNorm2d(out_channels)
        uprelu = nn.ReLU(True)
        upnorm = nn.BatchNorm2d(in_channels)

        if outermost:
            upconv = nn.ConvTranspose2d(out_channels * 2, in_channels, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(out_channels, in_channels, kernel_size=4, stride=2, padding=1, bias=False)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(out_channels * 2, in_channels, kernel_size=4, stride=2, padding=1, bias=False)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            if use_dropout:
                up += [nn.Dropout(0.5)]
            model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)

class GeneratorUNet(nn.Module):
    def __init__(self, input_channels=1, output_channels=2, num_down_blocks=8, base_filter_num=64):
        super(GeneratorUNet, self).__init__()
        # Building the U-Net model from inside out
        unet_block = UnetBlock(base_filter_num * 8, base_filter_num * 8, innermost=True)
        for _ in range(num_down_blocks - 5):
            unet_block = UnetBlock(base_filter_num * 8, base_filter_num * 8, submodule=unet_block, use_dropout=True)
        current_filter_num = base_filter_num * 8
        for _ in range(3):
            unet_block = UnetBlock(current_filter_num // 2, current_filter_num, submodule=unet_block)
            current_filter_num //= 2
        self.model = UnetBlock(output_channels, current_filter_num, input_channels=input_channels, submodule=unet_block, outermost=True)

    def forward(self, x):
        return self.model(x)
    
class PatchDiscriminator(nn.Module):
    def __init__(self, input_channels, num_filters=64, num_layers=3):
        super(PatchDiscriminator, self).__init__()

        layers = [self.get_layers(input_channels, num_filters, norm=False)]
        out_filters = num_filters

        for i in range(1, num_layers):
            in_filters = out_filters
            out_filters = num_filters * (2 ** i)
            if i == num_layers - 1:  # Last layer
                layers += [self.get_layers(in_filters, out_filters, s=1, norm=True)]
            else:
                layers += [self.get_layers(in_filters, out_filters, s=2, norm=True)]

        layers += [self.get_layers(out_filters, 1, s=1, norm=False, act=False)]  # Output layer

        self.model = nn.Sequential(*layers)

    def get_layers(self, in_filters, out_filters, k=4, s=2, p=1, norm=True, act=True):
        layers = [nn.Conv2d(in_filters, out_filters, k, s, p, bias=not norm)]
        if norm:
            layers += [nn.BatchNorm2d(out_filters)]
        if act:
            layers += [nn.LeakyReLU(0.2, inplace=True)]
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

