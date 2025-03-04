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

state_dict = torch.load('senior-design-proj/Unet-2/unet_v2.pth', map_location=torch.device('cpu'))
model = UNet(num_classes=15)
model.load_state_dict(state_dict)

criterion = nn.CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.eval()

if __name__ == "__main__":
    with torch.no_grad():
        val_loss = 0.0
        for images, labels in tqdm(valid_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)

        val_loss /= len(valid_loader.dataset)
        print(f"Validation Loss: {val_loss:.4f}")


# Validation Loss: 0.0338