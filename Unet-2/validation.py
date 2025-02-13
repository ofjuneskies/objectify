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

state_dict = torch.load('senior-design-proj/Unet-2/unet_epoch_100_2.pth', map_location=torch.device('cpu'))
model = UNet(num_classes=15)
model.load_state_dict(state_dict)

criterion = nn.CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.eval()

# image_path = "dataset/images/validation/image2395.png"
image_path = "dataset/images/train/image1.png"
image = Image.open(image_path).convert("RGB")
img_tensor = transform(image)
img_tensor = img_tensor[None, :, :, :]

output = model(img_tensor)
output = output.squeeze()
output = output.detach().numpy()

# np.save("output.npy", output)


# output = np.load("senior-design-proj/Unet/output2.npy")

output_t = torch.from_numpy(output)
probabilities = torch.softmax(output_t, dim=0)

# Compute confidence (max probability) for each pixel
confidence = torch.max(probabilities, dim=0)[0].numpy()

# Generate segmentation mask
segmentation_mask = np.argmax(probabilities.numpy(), axis=0)

# For thresholding instead (uncomment below):
threshold = 0.9
alpha = np.where(confidence >= threshold, 0.5, 0.0)  # Hard transparency cutoff

plt.imshow(image)
plt.imshow(segmentation_mask, cmap="jet", alpha=alpha)  # Apply alpha mask
plt.colorbar()
plt.axis("off")
plt.show()
