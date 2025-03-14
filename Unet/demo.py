import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from unet import UNet
import os

def segment_image(image_path):
    # Set up transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
        
    state_dict = torch.load('/Users/neeraj/Library/CloudStorage/OneDrive-Personal/UCI/Senior Design/senior-design-proj/Unet/unet_v7.pth', map_location=torch.device('cpu'))
    model = UNet(num_classes=16)
    model.load_state_dict(state_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return

    # Load and process input image
    try:
        input_image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image: {e}")
        return
        
    img_tensor = transform(input_image)
    img_tensor = img_tensor.unsqueeze(0).to(device)

    # Get model prediction
    with torch.no_grad():
        output = model(img_tensor)
        output = output.squeeze().cpu()
        probabilities = torch.softmax(output, dim=0)
        predicted_mask = torch.argmax(probabilities, dim=0).numpy()

    # Create visualization - only showing the predicted segmentation
    plt.figure(figsize=(10, 8))

    # Plot predicted segmentation overlaid on the input image
    predicted_alpha = np.where(predicted_mask == 15, 0.0, 0.7)  # Make background transparent
    plt.imshow(input_image)
    predicted_seg = plt.imshow(predicted_mask, cmap="jet", alpha=predicted_alpha)
    plt.title("Segmentation Result")
    plt.axis("off")

    # Add a colorbar
    cbar = plt.colorbar(predicted_seg, orientation='horizontal', fraction=0.046, pad=0.04)
    cbar.set_label('Class ID')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Prompt for image filepath
    image_path = input("Enter the path to the image file: ")
    segment_image(image_path)
