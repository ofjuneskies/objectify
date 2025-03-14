import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import ForgeDataset
from unet import UNet
# timer
import time

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

valid_images_folder = '/Users/neeraj/Library/CloudStorage/OneDrive-Personal/UCI/Senior Design/dataset/images/validation'
valid_labels_folder = '/Users/neeraj/Library/CloudStorage/OneDrive-Personal/UCI/Senior Design/dataset/labels/validation'
valid_dataset = ForgeDataset(images_folder=valid_images_folder, labels_folder=valid_labels_folder, transform=transform)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)

state_dict = torch.load('/Users/neeraj/Library/CloudStorage/OneDrive-Personal/UCI/Senior Design/senior-design-proj/Unet/unet_v7.pth', map_location=torch.device('cpu'))
model = UNet(num_classes=16)
model.load_state_dict(state_dict)

w = torch.tensor([0.902, 0.796, 0.851, 0.801, 0.866, 0.787, 0.848, 0.844, 0.426, 0.872, 1.0, 0.846, 0.826, 0.418, 0.207, 0.02])
criterion = nn.CrossEntropyLoss(weight=w)
# criterion = nn.CrossEntropyLoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.eval()

if __name__ == "__main__":
    with torch.no_grad():
        start = time.time()
        for images, labels in tqdm(valid_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
        end = time.time()
        print(f"Time taken: {end - start:.2f}s for {len(valid_loader)} images")
        print(f"FPS: {len(valid_loader) / (end - start):.2f}")


# unet_v1 - Validation Loss: 0.0303
# unet_v2 - Validation Loss: 0.0338
# unet_v3 - Validation Loss: 0.1006
# unet_v4 - Validation Loss: 0.0258
# unet_v5 - Validation Loss: 0.0648
# unet_v6 - Validation Loss: 0.0251