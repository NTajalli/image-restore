from model import *
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os
from torch import optim

# Adjusted for Lab color space processing
def train(generator, discriminator, dataloader, optimizer_G, optimizer_D, criterion, L1_loss, L1_lambda, epochs, device): 
    generator.to(device)
    discriminator.to(device)

    for epoch in range(epochs):
        for i, data in enumerate(dataloader):
            L = data['L'].to(device)  # Shape: [batch_size, 1, height, width]
            ab = data['ab'].to(device)
            vintage = data['vintage'].to(device)

            # Correctly reshape L to ensure it's a 4D tensor
            L = L.squeeze().unsqueeze(1)

            # Train Generator
            optimizer_G.zero_grad()
            gen_ab = generator(L)

            # Concatenate L channel with fake ab channels
            fake_images_lab = torch.cat((L, gen_ab), 1)

            # Determine the size of the output of discriminator
            patch_size = discriminator(torch.zeros_like(fake_images_lab)).size()[2:]
            valid = torch.ones((L.size(0), 1, *patch_size), device=device, requires_grad=False)
            fake = torch.zeros((L.size(0), 1, *patch_size), device=device, requires_grad=False)

            # Adversarial and L1 loss
            g_loss_adv = criterion(discriminator(fake_images_lab), valid)
            g_loss_L1 = L1_loss(gen_ab, ab) * L1_lambda
            g_loss = g_loss_adv + g_loss_L1
            g_loss.backward()
            optimizer_G.step()

            # Train Discriminator
            optimizer_D.zero_grad()
            real_images_lab = torch.cat((L, ab), 1)
            real_loss = criterion(discriminator(real_images_lab), valid)
            fake_loss = criterion(discriminator(fake_images_lab.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            # Logging
            print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

            # Save Images
            # Save Images
            batches_done = epoch * len(dataloader) + i
            if batches_done % 10 == 0:
                # Convert the vintage images to grayscale for visualization
                vintage_grayscale = vintage.data.mean(dim=1, keepdim=True)  # Averaging RGB channels to get grayscale

                # Convert the generated ab channels back to RGB
                gen_rgb = lab_to_rgb(L, gen_ab.data)  # This function needs to be implemented

                # Convert the real ab channels back to RGB
                real_rgb = lab_to_rgb(L, ab.data)  # This function needs to be implemented

                # Save the images
                save_image(vintage_grayscale, f"images/{batches_done}_vintage_grayscale.png", nrow=5, normalize=True)
                save_image(gen_rgb, f"images/{batches_done}_generated_rgb.png", nrow=5, normalize=True)
                save_image(real_rgb, f"images/{batches_done}_real_rgb.png", nrow=5, normalize=True)

def lab_to_rgb(L, ab):
    """
    Converts a batch of images from L*a*b* color space to RGB.
    Assumes L is in the range [0, 1] and a, b are in the range [-1, 1] if they were normalized.
    """
    # Denormalize L channel to [0, 100]
    L = (L * 100)
    # Denormalize a and b channels to the range [-127, 128]
    ab = (ab + 1) * 127.5

    # Stack to create the L*a*b* image
    Lab = torch.cat([L, ab], dim=1)
    Lab = Lab.detach().cpu().permute(0, 2, 3, 1).numpy()

    # Convert to BGR
    bgr_imgs = [cv2.cvtColor(lab_img.astype(np.float32), cv2.COLOR_LAB2BGR) for lab_img in Lab]
    bgr_imgs = np.stack(bgr_imgs, axis=0)

    # Convert BGR to RGB and make a contiguous copy of the array to ensure positive stride
    rgb_imgs = np.ascontiguousarray(bgr_imgs[..., ::-1])

    # Convert numpy arrays to tensors and permute back to BxCxHxW
    rgb_imgs = torch.from_numpy(rgb_imgs).permute(0, 3, 1, 2)
    
    # The values should already be in the correct range, so no clamping is necessary
    return rgb_imgs

                
# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
n_epochs = 50  # Number of epochs
lr = 0.0002    # Learning rate
L1_lambda = 100  # Weight for L1 loss

# Initialize models
generator = GeneratorUNet()
discriminator = Discriminator()

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

# Loss functions
criterion = nn.BCEWithLogitsLoss()
L1_loss = nn.L1Loss()

# Dataset and DataLoader
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    # No normalization should be applied as it's handled differently for Lab channels
])
vintage_dir = './vintage_images'
color_dir = './downloaded_images'
dataset = CustomImageDataset(vintage_dir=vintage_dir, color_dir=color_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Create a directory to save generated images
os.makedirs('images', exist_ok=True)

# Start training
train(generator, discriminator, dataloader, optimizer_G, optimizer_D, criterion, L1_loss, L1_lambda, n_epochs, device)
