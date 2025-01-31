import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import ForgeDataset
from unet import UNet
from process import get_img_label_tensor
from PIL import Image
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

state_dict = torch.load('senior-design-proj/Unet/unet_epoch_10.pth', map_location=torch.device('cpu'))
model = UNet(num_classes=15)
model.load_state_dict(state_dict)

criterion = nn.CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.eval()

# image_path = "yolo/images/validation/image2395.png"
image_path = "yolo/images/train/image1.png"
image = Image.open(image_path).convert("RGB")
img_tensor = transform(image)
img_tensor = img_tensor[None, :, :, :]

output = model(img_tensor)
output = output.squeeze()
output = output.detach().numpy()
print(output.shape)

np.save("output.npy", output)
# output = np.load("output.npy")
# output = output.squeeze()



'''
valid_images_folder = 'yolo/images/validation'
valid_labels_folder = 'yolo/labels/validation'
valid_dataset = ForgeDataset(images_folder=valid_images_folder, labels_folder=valid_labels_folder, transform=transform)
valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False, num_workers=4)

with torch.no_grad():
    val_loss = 0.0
    for images, labels in tqdm(valid_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        val_loss += loss.item() * images.size(0)

    val_loss /= len(valid_loader)
    print(f"Validation Loss: {val_loss:.4f}")
'''