import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Import YOLO model
from ultralytics import YOLO
from dataset import ForgeDataset

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
model = YOLO("senior-design-proj/Yolo/yolo11seg-100 results/yolo11n-segtrained-100.pt")
model.eval()

def generate_confusion_matrix(num_classes):
    # Initialize confusion matrix
    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    with torch.no_grad():
        for images, labels in tqdm(valid_loader):
            # Get the PIL image for YOLO inference
            # Convert tensor back to PIL image for YOLO
            img_tensor = images[0]
            img_np = img_tensor.numpy().transpose(1, 2, 0)
            # Denormalize
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_np = std * img_np + mean
            img_np = np.clip(img_np, 0, 1)
            img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
            
            # Get ground truth mask
            gt_mask = torch.argmax(labels.squeeze(), dim=0).cpu().numpy()
            
            # Run YOLO inference
            output = model(img_pil)
            
            # Process YOLO segmentation output
            masks = output[0].masks.data if output[0].masks is not None else None
            class_ids = output[0].boxes.cls if output[0].boxes is not None else None
            
            if masks is not None and class_ids is not None:
                segmentation = torch.zeros((num_classes, 640, 640), dtype=torch.float32)
                
                for i in range(len(class_ids)):
                    class_id = int(class_ids[i].item())
                    mask = masks[i]
                    segmentation[class_id] += mask
                
                # Get predicted class for each pixel
                pred_mask = torch.argmax(segmentation, dim=0).cpu().numpy()
                
                # Flatten the masks to 1D arrays
                pred_flat = pred_mask.flatten()
                gt_flat = gt_mask.flatten()
                
                # Update confusion matrix
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
    plt.title('YOLO Confusion Matrix (Normalized)')
    plt.tight_layout()
    plt.savefig('yolo_confusion_matrix_normalized.png')
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
        print(f"{class_names[i]}: {class_accuracy[i]:.4f}")
    
    # Calculate overall accuracy
    overall_accuracy = np.sum(np.diag(conf_matrix)) / np.sum(conf_matrix)
    print(f"\nOverall Accuracy: {overall_accuracy:.4f}")
