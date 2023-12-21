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

            print("Shape of L before reshape:", L.shape)  # Debug print

            # Correctly reshape L to ensure it's a 4D tensor: [batch_size, 1, height, width]
            L = L.squeeze()  # Removes unnecessary dimensions
            L = L.unsqueeze(1)  # Adds the channel dimension back

            print("Shape of L after reshape:", L.shape)  # Debug print

            # Adversarial ground truths
            valid = torch.ones((L.size(0), 1), device=device, requires_grad=False)
            fake = torch.zeros((L.size(0), 1), device=device, requires_grad=False)

            # Train Generator
            optimizer_G.zero_grad()
            gen_ab = generator(vintage)  # Shape: [batch_size, 2, height, width]

            print("Shape of gen_ab:", gen_ab.shape)  # Debug print

            # Concatenate L channel with fake ab channels
            try:
                fake_images_lab = torch.cat((L, gen_ab), 1)
            except RuntimeError as e:
                print("Error during concatenation:", e)
                print("Shape of L:", L.shape)
                print("Shape of gen_ab:", gen_ab.shape)
                raise e

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
            batches_done = epoch * len(dataloader) + i
            if batches_done % 10 == 0:
                sample_images = torch.cat((vintage.data, gen_ab.data, ab.data), -1)
                save_image(sample_images, f"images/{batches_done}.png", nrow=5, normalize=True)

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
criterion = nn.BCELoss()
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
