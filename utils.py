import time

import numpy as np
from skimage.color import rgb2lab, lab2rgb
import matplotlib.pyplot as plt
import warnings
import torch


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.count, self.avg, self.sum = [0.] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += count * val
        self.avg = self.sum / self.count


def create_loss_meters():
    loss_D_fake = AverageMeter()
    loss_D_real = AverageMeter()
    loss_D_perceptual = AverageMeter()
    loss_D = AverageMeter()
    loss_G_GAN = AverageMeter()
    loss_G_L1 = AverageMeter()
    loss_G_perceptual = AverageMeter()
    loss_G = AverageMeter()

    return {'loss_D_fake': loss_D_fake,
            'loss_D_real': loss_D_real,
            'loss_D_perceptual': loss_D_perceptual,
            'loss_D': loss_D,
            'loss_G_GAN': loss_G_GAN,
            'loss_G_L1': loss_G_L1,
            'loss_G_perceptual': loss_G_perceptual,
            'loss_G': loss_G}


def update_losses(model, loss_meter_dict, count):
    for loss_name, loss_meter in loss_meter_dict.items():
        loss = getattr(model, loss_name)
        loss_meter.update(loss.item(), count=count)


def lab_to_rgb(L, ab, is_tensor=False):
    """
    Takes a batch of images
    """

    L = (L + 1.) * 50.
    ab = ab * 110.
    
    if is_tensor:
        Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().detach().numpy()
    else:
        Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
        
    rgb_imgs = []
    for img in Lab:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            img_rgb = lab2rgb(img)
        rgb_imgs.append(img_rgb)
    return np.stack(rgb_imgs, axis=0)


def visualize(model, data, save=True):
    model.net_G.eval()
    with torch.no_grad():
        model.setup_input(data)
        model.forward()
    model.net_G.train()
    fake_color = model.fake_color.detach()
    real_color = model.ab
    L = model.L
    fake_imgs = lab_to_rgb(L, fake_color)
    real_imgs = lab_to_rgb(L, real_color)
    fig = plt.figure(figsize=(15, 8))
    for i in range(5):
        ax = plt.subplot(3, 5, i + 1)
        ax.imshow(L[i][0].cpu(), cmap='gray')
        ax.axis("off")
        ax = plt.subplot(3, 5, i + 1 + 5)
        ax.imshow(fake_imgs[i])
        ax.axis("off")
        ax = plt.subplot(3, 5, i + 1 + 10)
        ax.imshow(real_imgs[i])
        ax.axis("off")
    plt.show()
    if save:
        fig.savefig(f"colorizationATTENTION_{time.time()}.png")
        
def visualize_rgb(fake_rgb, real_rgb, save=True, filename_suffix=''):
    """
    Visualizes and saves real and fake RGB images.

    :param fake_rgb: Fake RGB images generated by the model.
    :param real_rgb: Real RGB images.
    :param save: Whether to save the images.
    :param filename_suffix: Suffix for the filename when saving images.
    """
    fig = plt.figure(figsize=(10, 4))
    for i in range(min(5, fake_rgb.size(0))):  # Display up to 5 images
        ax = plt.subplot(2, 5, i + 1)
        ax.imshow(fake_rgb[i].permute(1, 2, 0).cpu().numpy())
        ax.axis("off")
        ax = plt.subplot(2, 5, i + 6)
        ax.imshow(real_rgb[i].permute(1, 2, 0).cpu().numpy())
        ax.axis("off")
    plt.show()
    if save:
        fig.savefig(f"colorizationATTENTION_{filename_suffix}_{time.time()}.png")



def log_results(loss_meter_dict):
    for loss_name, loss_meter in loss_meter_dict.items():
        print(f"{loss_name}: {loss_meter.avg:.5f}")