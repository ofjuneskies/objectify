import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from dataset import ForgeDataset
from unet import UNet
from ultralytics import YOLO
from process import get_img_label_tensor

# Set up transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Define paths
# image_path = "dataset/images/validation/image2400.png"
# label_path = "dataset/labels/validation/image2400.txt"
image_path = "dataset/images/validation/image2436.png"
label_path = "dataset/labels/validation/image2436.txt"

# Load and process input image
input_image = Image.open(image_path).convert("RGB")
img_tensor = transform(input_image)
img_tensor = img_tensor.unsqueeze(0)

# Get expected segmentation from label
_, label_tensor = get_img_label_tensor(1, image_path, label_path, transform=None)
expected_mask = torch.argmax(label_tensor, dim=0).numpy()

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create a figure with subplots for all models plus input and expected output
fig, axes = plt.subplots(3, 3, figsize=(20, 15))
axes = axes.flatten()

# Plot input image
axes[0].imshow(input_image)
axes[0].set_title("Input Image", fontsize=14)
axes[0].axis("off")

# Plot expected segmentation
expected_alpha = np.where(expected_mask == 15, 0.0, 0.7)
axes[1].imshow(input_image)
expected_seg = axes[1].imshow(expected_mask, cmap="jet", alpha=expected_alpha, vmin=0, vmax=15)
axes[1].set_title("Ground Truth", fontsize=14)
axes[1].axis("off")

# Get YOLO prediction
yolo_model = YOLO("senior-design-proj/Yolo/yolo11seg-100 results/yolo11n-segtrained-100.pt")
yolo_model.to(device)

with torch.no_grad():
    output = yolo_model(input_image)
    masks = output[0].masks.data
    class_ids = output[0].boxes.cls
    segmentation = torch.zeros((15, 640, 640), dtype=torch.float32)
    for i in range(len(class_ids)):
        class_id = int(class_ids[i].item())
        mask = masks[i]
        segmentation[class_id] += mask
    binary_segmentation = (segmentation > 0.5).float()
    yolo_predicted_mask = torch.argmax(segmentation, dim=0).numpy()

# Plot YOLO prediction
yolo_alpha = np.where(yolo_predicted_mask == 0, 0.0, 0.7)
axes[2].imshow(input_image)
yolo_seg = axes[2].imshow(yolo_predicted_mask, cmap="jet", alpha=yolo_alpha, vmin=0, vmax=15)
axes[2].set_title("YOLO Prediction", fontsize=14)
axes[2].axis("off")

# Load each UNet model and get predictions
unet_predictions = []
for i in range(3, 9):
    model_path = f'senior-design-proj/Unet/unet_v{i}.pth'
    # Load model
    state_dict = torch.load(model_path, map_location=device)
    model = UNet(num_classes=16)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # Get model prediction
    with torch.no_grad():
        output = model(img_tensor.to(device))
        output = output.squeeze().cpu()
        probabilities = torch.softmax(output, dim=0)
        predicted_mask = torch.argmax(probabilities, dim=0).numpy()
        unet_predictions.append(predicted_mask)
    
    # Plot predicted segmentation
    predicted_alpha = np.where(predicted_mask == 15, 0.0, 0.7)
    axes[i].imshow(input_image)
    pred_seg = axes[i].imshow(predicted_mask, cmap="jet", alpha=predicted_alpha, vmin=0, vmax=15)
    axes[i].set_title(f"UNet v{i} Prediction", fontsize=14)
    axes[i].axis("off")

plt.tight_layout()
plt.savefig("model_comparison.png", dpi=300, bbox_inches='tight')
plt.show()
