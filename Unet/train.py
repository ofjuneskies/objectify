import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import ForgeDataset
from unet import UNet
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Define the training function
def train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs, device, scheduler):
    f = open("training.txt", "a")
    model.to(device)

    for epoch in range(num_epochs):
        running_loss = 0.0
        model.train()

        # Iterate through the dataloader
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        # Calculate average loss for the epoch
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")
        f.write(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}\n")

        # Check the model performance on the validation set
        model.eval() # Set the model to evaluation mode
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in valid_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)

            val_loss /= len(valid_loader.dataset)
            print(f"Validation Loss: {val_loss:.4f}")
            f.write(f"Validation Loss: {val_loss:.4f}\n")

        scheduler.step(val_loss)
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")

        # Save the model after each epoch
        torch.save(model.state_dict(), f"unet_epoch_{epoch + 1}.pth")

    print("Training complete.")
    f.close()

model = UNet(num_classes=16)

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Freeze the encoder layers
for param in model.encoder1.parameters():
    param.requires_grad = False
for param in model.encoder2.parameters():
    param.requires_grad = False
for param in model.encoder3.parameters():
    param.requires_grad = False
for param in model.encoder4.parameters():
    param.requires_grad = False
for param in model.encoder5.parameters():
    param.requires_grad = False

# Define the loss function and optimizer
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.05, patience=15)

w = torch.tensor([0.902, 0.796, 0.851, 0.801, 0.866, 0.787, 0.848, 0.844, 0.426, 0.872, 1.0, 0.846, 0.826, 0.418, 0.207, 0.02]).to(device)
criterion = nn.CrossEntropyLoss(weight=w)

# Training parameters
num_epochs = 100

# Transform for data augmentation
transform = transforms.Compose([
    transforms.RandomApply([transforms.ColorJitter(contrast=0.5)], p=1),
    transforms.RandomApply([transforms.ColorJitter(brightness=0.5)], p=1),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=(21, 21))], p=1),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # Normalize the image
])

# Create the dataset and dataloader
train_images_folder = './dataset/images/train'
train_labels_folder = './dataset/labels/train'
train_dataset = ForgeDataset(images_folder=train_images_folder, labels_folder=train_labels_folder, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)

valid_images_folder = './dataset/images/validation'
valid_labels_folder = './dataset/labels/validation'
valid_dataset = ForgeDataset(images_folder=valid_images_folder, labels_folder=valid_labels_folder, transform=transform)
valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False, num_workers=4)

# Train the model
train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs, device, scheduler)