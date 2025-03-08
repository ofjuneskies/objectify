import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming these imports are available as in the original file
from dataset import ForgeDataset
from unet import UNet

# Set up transforms and data loading
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

valid_images_folder = 'dataset/images/validation'
valid_labels_folder = 'dataset/labels/validation'
valid_dataset = ForgeDataset(images_folder=valid_images_folder, labels_folder=valid_labels_folder, transform=transform)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)

# Set up device and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state_dict = torch.load('senior-design-proj/Unet/unet_v3.pth', map_location=torch.device(device))
model = UNet(num_classes=16).to(device)
model.load_state_dict(state_dict)
model.eval()

def generate_confusion_matrix(num_classes):
    # Initialize confusion matrix
    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    with torch.no_grad():
        for images, labels in tqdm(valid_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            output = model(images)
            output = output.squeeze()
            
            # Convert output to probabilities and get predicted class
            output = output.detach().cpu().numpy()
            output_t = torch.from_numpy(output)
            probabilities = torch.softmax(output_t, dim=0)
            pred_mask = np.argmax(probabilities.numpy(), axis=0)
            
            # Get ground truth mask - assuming labels are one-hot encoded
            # Convert to class indices
            gt_mask = torch.argmax(labels.squeeze(), dim=0).cpu().numpy()
            
            # Update confusion matrix
            # Flatten the masks to 1D arrays
            pred_flat = pred_mask.flatten()
            gt_flat = gt_mask.flatten()
            
            # Update confusion matrix
            # We only consider classes 0-14 (15 classes total)
            mask = (gt_flat < num_classes) & (pred_flat < num_classes)
            curr_conf_matrix = confusion_matrix(
                gt_flat[mask], 
                pred_flat[mask], 
                labels=range(num_classes)
            )
            conf_matrix += curr_conf_matrix
    
    return conf_matrix

def plot_confusion_matrix(conf_matrix, class_names=None):
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(conf_matrix))]
    
    # Convert to percentages by dividing by row sums (true class totals)
    conf_matrix_percent = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    # Replace NaN with 0 (in case there are rows with sum=0)
    conf_matrix_percent = np.nan_to_num(conf_matrix_percent)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        conf_matrix_percent,
        annot=True,
        fmt='.2f',  # Changed format to show float values with 2 decimal places
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Normalized) v3')
    plt.tight_layout()
    plt.savefig('confusion_matrix_normalized v3.png')
    plt.show()

if __name__ == "__main__":
    # Generate confusion matrix
    conf_matrix = generate_confusion_matrix(num_classes=16)
    
    class_names = ["mannequin", "suitcase", "tennisracket", "boat", "stopsign", "plane", "baseballbat", "bus", "mattress", "skis", "umbrella", "snowboard", "motorcycle", "car", "sportsball", "background"]
    
    # Plot and save confusion matrix
    plot_confusion_matrix(conf_matrix, class_names)
    
    # Print some metrics from the confusion matrix
    print("Confusion Matrix:")
    print(conf_matrix)
    
    # Calculate per-class accuracy
    class_accuracy = np.zeros(15)
    for i in range(15):
        class_accuracy[i] = conf_matrix[i, i] / np.sum(conf_matrix[i, :]) if np.sum(conf_matrix[i, :]) > 0 else 0
    
    print("\nPer-class Accuracy:")
    for i in range(15):
        print(f"Class {i}: {class_accuracy[i]:.4f}")
    
    # Calculate overall accuracy
    overall_accuracy = np.sum(np.diag(conf_matrix)) / np.sum(conf_matrix)
    print(f"\nOverall Accuracy: {overall_accuracy:.4f}")
