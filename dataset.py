import numpy as np
from PIL import Image
from skimage.color import rgb2lab, lab2rgb

import torch
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os 


class CustomImageDataset(Dataset):
    def __init__(self, vintage_dir, color_dir, filenames, transform=None):
        self.vintage_dir = vintage_dir
        self.color_dir = color_dir
        self.transform = transform
        self.image_filenames = filenames

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        vintage_img_name = os.path.join(self.vintage_dir, self.image_filenames[idx])
        color_img_name = os.path.join(self.color_dir, self.image_filenames[idx])

        vintage_image = Image.open(vintage_img_name).convert('RGB')
        color_image = Image.open(color_img_name).convert('RGB')

        # Apply transformations
        if self.transform:
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

        return {'vintage': vintage_image, 'color': color_image, 'L': L, 'ab': ab}
