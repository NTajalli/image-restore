import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
import os
import numpy as np

class CustomImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            # Assuming input images are RGB, so in_channels = 3
            nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()  # Output is an RGB image
        )

    def forward(self, img):
        return self.model(img)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=2, padding=1),
                     nn.LeakyReLU(0.2, inplace=True),
                     nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(3, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # Dynamically determine the flatten size
        self.determine_flatten_size()

        self.adv_layer = nn.Sequential(
            nn.Linear(self.flatten_size, 1),
            nn.Sigmoid()
        )

    def determine_flatten_size(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 256, 256)
            dummy_output = self.model(dummy_input)
            self.flatten_size = int(np.prod(dummy_output.size()[1:]))
            print(f"Flatten size: {self.flatten_size}")  # Print the flatten size

    def forward(self, img):
        print(f"Input shape: {img.shape}")  # Print the input shape
        out = self.model(img)
        print(f"Shape after conv layers: {out.shape}")  # Print shape after convolutional layers
        out = out.view(out.size(0), -1)  # Flatten the output for the linear layer
        print(f"Shape after flattening: {out.shape}")  # Print shape after flattening
        validity = self.adv_layer(out)
        print(f"Output shape: {validity.shape}")  # Print the final output shape
        return validity


