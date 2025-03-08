import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from dataset import ForgeDataset
from ultralytics import YOLO
from process import get_img_label_tensor

# Set up transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Load model
model = YOLO("senior-design-proj/Yolo/yolo11seg-100 results/yolo11n-segtrained-100.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.eval()

# Define paths
image_path = "dataset/images/validation/image2400.png"
label_path = "dataset/labels/validation/image2400.txt"

# Load and process input image
input_image = Image.open(image_path).convert("RGB")
img_tensor = transform(input_image)
img_tensor = img_tensor.unsqueeze(0).to(device)

# Get expected segmentation from label
_, label_tensor = get_img_label_tensor(1, image_path, label_path, transform=None)
expected_mask = torch.argmax(label_tensor, dim=0).numpy()

# Get model prediction
with torch.no_grad():
    print(input_image)
    output = model(input_image)
    masks = output[0].masks.data
    class_ids = output[0].boxes.cls
    segmentation = torch.zeros((15, 640, 640), dtype=torch.float32)
    for i in range(len(class_ids)):
        class_id = int(class_ids[i].item())
        mask = masks[i]
        segmentation[class_id] += mask
    binary_segmentation = (segmentation > 0.5).float()
    predicted_mask = torch.argmax(segmentation, dim=0)

# Create visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot input image
axes[0].imshow(input_image)
axes[0].set_title("Input Image")
axes[0].axis("off")

# Plot expected segmentation
# Apply alpha mask to make background transparent
expected_alpha = np.where(expected_mask == 15, 0.0, 0.7)
axes[1].imshow(input_image)
expected_seg = axes[1].imshow(expected_mask, cmap="jet", alpha=expected_alpha, vmin=0, vmax=15)
axes[1].set_title("Expected Segmentation")
axes[1].axis("off")

# Plot predicted segmentation
predicted_alpha = np.where(predicted_mask == 0, 0.0, 0.7)
axes[2].imshow(input_image)
predicted_seg = axes[2].imshow(predicted_mask, cmap="jet", alpha=predicted_alpha, vmin=0, vmax=15)
axes[2].set_title("Predicted Segmentation")
axes[2].axis("off")

# Add a colorbar
cbar = fig.colorbar(predicted_seg, ax=axes, orientation='horizontal', fraction=0.02, pad=0.05)
cbar.set_label('Class ID')

plt.tight_layout()
plt.savefig("segmentation_comparison.png", dpi=300, bbox_inches='tight')
plt.show()
