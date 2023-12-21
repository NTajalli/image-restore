from model import *
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os
from torch import optim
from skimage.color import lab2rgb


# Adjusted for Lab color space processing
def train(generator, discriminator, dataloader, optimizer_G, optimizer_D, criterion, L1_loss, L1_lambda, epochs, device): 
    generator.to(device)
    discriminator.to(device)
    os.makedirs('model_snapshots', exist_ok=True)

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
            if batches_done % 30 == 0:
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
                
        # Save generator and discriminator state
        torch.save(generator.state_dict(), f'model_snapshots/generator_epoch_{epoch}.pth')
        torch.save(discriminator.state_dict(), f'model_snapshots/discriminator_epoch_{epoch}.pth')

        print(f"Saved model snapshots for epoch {epoch}")

        # You may also want to save snapshots of your optimizers' states
        torch.save(optimizer_G.state_dict(), f'model_snapshots/optimizerG_epoch_{epoch}.pth')
        torch.save(optimizer_D.state_dict(), f'model_snapshots/optimizerD_epoch_{epoch}.pth')

def lab_to_rgb(L, ab):
    """
    Converts a batch of images from L*a*b* color space to RGB.
    Assumes L is in the range [-1, 1] and a, b are in the range [-1, 1].
    """
    L = (L + 1.) * 50.  # Rescale L channel to [0, 100]
    ab = ab * 110.  # Rescale ab channels to [-128, 127]

    colorized_images = []

    for i in range(L.shape[0]):
        L_img = L[i].cpu().numpy()
        ab_img = ab[i].cpu().numpy()

        # Reshape L to match the dimensions of ab
        L_img = L_img.squeeze()  # Remove any singleton dimensions

        # Stack L and ab channels
        Lab_img = np.stack((L_img, ab_img[0], ab_img[1]), axis=-1)  # Stack along the last dimension

        # Convert Lab to RGB
        rgb_img = lab2rgb(Lab_img)

        # Convert the RGB image to a tensor and add to the list
        colorized_images.append(torch.from_numpy(rgb_img).permute(2, 0, 1))

    return torch.stack(colorized_images)


     
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

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),  # Data augmentation
    transforms.RandomRotation(10),  # Random rotation +/- 10 degrees
    transforms.ToTensor(),
])

# For Validation (without data augmentation)
val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Usage
vintage_dir = './vintage_images'
color_dir = './downloaded_images'

# For training dataset
train_dataset = CustomImageDataset(vintage_dir=vintage_dir, color_dir=color_dir, transform=train_transform, is_train=True)

# For validation dataset
val_dataset = CustomImageDataset(vintage_dir=vintage_dir, color_dir=color_dir, transform=val_transform, is_train=False)

# Dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Create a directory to save generated images
os.makedirs('images', exist_ok=True)

# Start training
train(generator, discriminator, train_dataloader, optimizer_G, optimizer_D, criterion, L1_loss, L1_lambda, n_epochs, device)
