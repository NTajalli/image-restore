from model import *


def train(generator, discriminator, vintage_dataloader, original_dataloader, optimizer_G, optimizer_D, criterion, epochs, device):
    generator.to(device)
    discriminator.to(device)
    
    for epoch in range(epochs):
        # Use itertools.zip_longest if dataloaders have different lengths
        for i, (vintage_imgs, real_imgs) in enumerate(zip(vintage_dataloader, original_dataloader)):

            # Adversarial ground truths
            valid = torch.ones(real_imgs.size(0), 1, requires_grad=False).to(device)
            fake = torch.zeros(vintage_imgs.size(0), 1, requires_grad=False).to(device)

            real_imgs = real_imgs.to(device)
            vintage_imgs = vintage_imgs.to(device)

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()

            # Generate a batch of images
            gen_imgs = generator(vintage_imgs)
            

            # Loss measures generator's ability to fool the discriminator
            g_loss = criterion(discriminator(gen_imgs), valid)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = criterion(discriminator(real_imgs), valid)
            fake_loss = criterion(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(vintage_dataloader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

            batches_done = epoch * len(vintage_dataloader) + i
            if batches_done % 10 == 0:
                save_image(gen_imgs.data[:25], f"images/{batches_done}.png", nrow=5, normalize=True)




# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
n_epochs = 50  # Number of epochs
lr = 0.0002    # Learning rate

# Initialize models
generator = Generator()
discriminator = Discriminator()

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

# Loss function
criterion = nn.BCELoss()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Create the dataset and dataloader
# Dataset and Dataloader for vintage images (Generator input)
vintage_dataset = CustomImageDataset(image_dir='vintage_images', transform=transform)
vintage_dataloader = DataLoader(vintage_dataset, batch_size=64, shuffle=True)

# Dataset and Dataloader for original images (Discriminator real samples)
original_dataset = CustomImageDataset(image_dir='downloaded_images', transform=transform)
original_dataloader = DataLoader(original_dataset, batch_size=64, shuffle=True)


# Create a directory to save generated images
os.makedirs('images', exist_ok=True)

train(generator, discriminator, vintage_dataloader, original_dataloader, optimizer_G, optimizer_D, criterion, n_epochs, device)

