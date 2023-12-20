from model import *


def train(generator, discriminator, dataloader, optimizer_G, optimizer_D, criterion, epochs, device):
    generator.to(device)
    discriminator.to(device)
    
    for epoch in range(epochs):
        for i, imgs in enumerate(dataloader):

            # Adversarial ground truths
            valid = torch.ones(imgs.size(0), 1, requires_grad=False).to(device)
            fake = torch.zeros(imgs.size(0), 1, requires_grad=False).to(device)

            # Configure input
            real_imgs = imgs.to(device)

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()

            # Sample noise as generator input
            z = torch.randn(imgs.size(0), 100).to(device)

            # Generate a batch of images
            gen_imgs = generator(z)

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

            print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")
            
            batches_done = epoch * len(dataloader) + i
            if batches_done % 400 == 0:
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
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Create the dataset and dataloader
dataset = CustomImageDataset(image_dir='downloaded_images', transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Create a directory to save generated images
os.makedirs('images', exist_ok=True)

train(generator, discriminator, dataloader, optimizer_G, optimizer_D, criterion, n_epochs, device)

