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
from matplotlib.colors import ListedColormap

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

class_names = {
    0: "mannequin",
    1: "suitcase",
    2: "tennisracket",
    3: "boat",
    4: "stopsign",
    5: "plane",
    6: "baseballbat",
    7: "bus",
    8: "mattress",
    9: "skis",
    10: "umbrella",
    11: "snowboard",
    12: "motorcycle",
    13: "car",
    14: "sportsball"
}

state_dict = torch.load('senior-design-proj/Unet/unet_v5.pth', map_location=torch.device('cpu'))
model = UNet(num_classes=16)
model.load_state_dict(state_dict)

criterion = nn.CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.eval()

# Create a custom colormap for the segmentation mask
num_classes = len(class_names)
colors = plt.cm.jet(np.linspace(0, 1, num_classes))  # Generate distinct colors
cmap = ListedColormap(colors)

# Define the image paths
image_paths = [
    "dataset/images/validation/image2400.png",
    "dataset/images/validation/image2410.png",
    "dataset/images/validation/image2420.png",
    "dataset/images/validation/image2430.png",
    "dataset/images/validation/image2520.png",
    "dataset/images/validation/image2530.png",
    "dataset/images/validation/image2470.png",
    "dataset/images/validation/image2490.png",
    "dataset/images/validation/image2500.png"
]

# Create a 3x3 grid of images
fig, axes = plt.subplots(3, 3, figsize=(10, 8))

for idx, image_path in enumerate(image_paths):
    # Load and process the image
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image)
    img_tensor = img_tensor[None, :, :, :]  # Add batch dimension

    # Get model output
    output = model(img_tensor)
    output = output.squeeze()  # Remove batch dimension
    output = output.detach().numpy()

    # Convert output to probabilities
    output_t = torch.from_numpy(output)
    probabilities = torch.softmax(output_t, dim=0)

    # Compute segmentation mask
    segmentation_mask = np.argmax(probabilities.numpy(), axis=0)

    # Display the image and segmentation mask
    row, col = divmod(idx, 3)
    axes[row, col].imshow(image)
    
    # Overlay the segmentation mask with transparency
    alpha = np.where(segmentation_mask == 15, 0.0, 0.5)  # Adjust transparency for class 15
    axes[row, col].imshow(segmentation_mask, cmap="jet", alpha=alpha)
    
    axes[row, col].axis("off")

# Add a legend outside the grid
legend_patches = [plt.Line2D([0], [0], color=colors[i], lw=4) for i in range(15)]
legend_labels = [class_names[i] for i in range(15)]

fig.legend(
    handles=legend_patches,
    labels=legend_labels,
    loc="center left",  # Position it to the left or adjust as needed
    title="Classes",
    bbox_to_anchor=(0.85, 0.5),  # Slightly outside the main figure but still visible
)

plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to make space for the legend
plt.show()