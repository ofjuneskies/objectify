import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import ForgeDataset
from unet import UNet

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

valid_images_folder = 'dataset/images/validation'
valid_labels_folder = 'dataset/labels/validation'
valid_dataset = ForgeDataset(images_folder=valid_images_folder, labels_folder=valid_labels_folder, transform=transform)
valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False, num_workers=4)

for i in range(1, 101):
    state_dict = torch.load(f'senior-design-proj/Unet/unet_epoch_{i}.pth', map_location=torch.device('cuda'))
    model = UNet(num_classes=16)
    model.load_state_dict(state_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    w = torch.tensor([0.902, 0.796, 0.851, 0.801, 0.866, 0.787, 0.848, 0.844, 0.426, 0.872, 1.0, 0.846, 0.826, 0.418, 0.207, 0.02]).to(device)
    criterion = nn.CrossEntropyLoss(weight=w)

    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)

        val_loss /= len(valid_loader.dataset)
        print(f"Validation Loss at Epoch {i}: {val_loss:.4f}")
