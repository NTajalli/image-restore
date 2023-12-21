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
    def __init__(self, in_channels, out_channels, submodule=None, outermost=False, innermost=False, use_dropout=False):
        super(UnetBlock, self).__init__()
        self.outermost = outermost
        downconv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        downrelu = nn.LeakyReLU(0.2, True)
        uprelu = nn.ReLU(True)
        upnorm = nn.BatchNorm2d(out_channels)

        if outermost:
            upconv = nn.ConvTranspose2d(out_channels * 2, out_channels, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(out_channels * 2, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
            down = [downrelu, downconv, nn.BatchNorm2d(out_channels)]
            up = [uprelu, upconv, upnorm]
            if use_dropout:
                up += [nn.Dropout(0.5)]
            model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        print(f"UnetBlock - Input shape: {x.shape}")
        
        if self.outermost:
            output = self.model(x)
            print(f"UnetBlock (Outermost) - Input shape: {x.shape}, Output shape: {output.shape}")
            return output
        else:
            output = self.model(x)
            concatenated_output = torch.cat([x, output], 1)
            print(f"UnetBlock - Input shape: {x.shape}, Output shape: {output.shape}, Concatenated shape: {concatenated_output.shape}")
            return concatenated_output


class GeneratorUNet(nn.Module):
    def __init__(self, input_channels=1, output_channels=2):
        super(GeneratorUNet, self).__init__()

        # Innermost layer
        unet_innermost = UnetBlock(512, 512, innermost=True)

        # Middle layers
        unet_middle_0 = UnetBlock(512, 512, submodule=unet_innermost, use_dropout=True)
        unet_middle_1 = UnetBlock(512, 512, submodule=unet_middle_0, use_dropout=True)
        unet_middle_2 = UnetBlock(256, 512, submodule=unet_middle_1)
        unet_middle_3 = UnetBlock(128, 256, submodule=unet_middle_2)
        unet_middle_4 = UnetBlock(64, 128, submodule=unet_middle_3)

        # Outermost layer
        self.model = UnetBlock(output_channels, 64, submodule=unet_middle_4, outermost=True)

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

