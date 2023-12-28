

from dataset import *
from models import MainModel
from utils import *
from tqdm import tqdm
import os

def train_val_split(vintage_dir, val_split=0.2):
    filenames = [f for f in os.listdir(vintage_dir) if os.path.isfile(os.path.join(vintage_dir, f))]
    filenames.sort()
    num_val_samples = int(val_split * len(filenames))
    val_filenames = filenames[:num_val_samples]
    train_filenames = filenames[num_val_samples:]
    return train_filenames, val_filenames


def train_model(model, train_dl, val_dl, epochs, display_every=200, snapshot_path='./model_snapshotsSELFATTENTION'):
    data = next(iter(val_dl)) # getting a batch for visualizing the model output after fixed intrvals
    os.makedirs(snapshot_path, exist_ok=True)
    for e in range(epochs):
        loss_meter_dict = create_loss_meters() # function returing a dictionary of objects to 
        i = 0                                  # log the losses of the complete network
        for data in tqdm(train_dl):
            model.setup_input(data) 
            model.optimize()
            update_losses(model, loss_meter_dict, count=data['L'].size(0)) # function updating the log objects
            i += 1
            if i % display_every == 0:
                print(f"\nEpoch {e+1}/{epochs}")
                print(f"Iteration {i}/{len(train_dl)}")
                log_results(loss_meter_dict) # function to print out the losses
                visualize(model, data) # function displaying the model's outputs
                
                # Save model snapshots
                snapshot_filename = f"model_epoch_{e+1}_iter_{i}.pth"
                snapshot_filepath = os.path.join(snapshot_path, snapshot_filename)
                torch.save(model.state_dict(), snapshot_filepath)
                print(f"Saved model snapshot to {snapshot_filepath}")

vintage_dir = './vintage_images'
color_dir = './downloaded_images'
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),  # Data augmentation
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
# Use the split function to get the train and validation filenames
train_filenames, val_filenames = train_val_split(vintage_dir, val_split=0.2)

# Create training and validation datasets
train_dataset = CustomImageDataset(vintage_dir, color_dir, train_filenames, transform=train_transform)
val_dataset = CustomImageDataset(vintage_dir, color_dir, val_filenames, transform=val_transform)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)

model = MainModel()

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = torch.nn.DataParallel(model)

train_model(model, train_dataloader, val_dataloader, 100)