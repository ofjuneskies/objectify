import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from process import get_img_label_tensor
from torchmetrics.segmentation import MeanIoU
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import ForgeDataset
from unet import UNet

from torch import randint


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

valid_images_folder = 'dataset/images/validation'
valid_labels_folder = 'dataset/labels/validation'
valid_dataset = ForgeDataset(images_folder=valid_images_folder, labels_folder=valid_labels_folder, transform=transform)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

state_dict = torch.load('senior-design-proj/Unet-2/unet_v2.pth', map_location=torch.device(device))
model = UNet(num_classes=15).to(device)
model.load_state_dict(state_dict)

model.eval()

def calculate_iou(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    iou = intersection / union if union != 0 else 0
    return iou

def calculate_metrics(pred_mask, gt_mask):
    """
    Calculate pixel-wise accuracy, precision, and recall.
    Handles edge cases where both pred_mask and gt_mask are empty (all zeros).
    """
    pred_flat = pred_mask.flatten()
    gt_flat = gt_mask.flatten()

    # True Positives (TP), False Positives (FP), False Negatives (FN)
    tp = np.sum((pred_flat == 1) & (gt_flat == 1))
    fp = np.sum((pred_flat == 1) & (gt_flat == 0))
    fn = np.sum((pred_flat == 0) & (gt_flat == 1))
    tn = np.sum((pred_flat == 0) & (gt_flat == 0))

    # Handle edge cases where there are no positive pixels in both pred and gt
    if tp + fp + fn == 0:
        precision = 1.0  # No false positives
        recall = 1.0     # No false negatives
    else:
        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0

    # Accuracy
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) != 0 else 0

    return accuracy, precision, recall

def calculate_map50(pred_mask, gt_mask):
    """
    Calculate mAP50 for a single prediction.
    Handles edge cases where both pred_mask and gt_mask are empty (all zeros).
    """
    if np.sum(gt_mask) == 0 and np.sum(pred_mask) == 0:
        # If both masks are empty, consider it a perfect match
        return 1.0
    elif np.sum(gt_mask) == 0 or np.sum(pred_mask) == 0:
        # If one is empty but the other is not, IoU is zero
        return 0.0
    else:
        iou = calculate_iou(pred_mask, gt_mask)
        return 1 if iou >= 0.5 else 0

if __name__ == "__main__":
    # tensor of 0's
    # pred_class_mask = torch.zeros((256, 256)).cpu().numpy()
    # gt_class_mask = torch.zeros((256, 256)).cpu().numpy()
    # acc, prec, rec = calculate_metrics(pred_class_mask, gt_class_mask)
    # map50_score = calculate_map50(pred_class_mask, gt_class_mask)
    # print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, mAP@50: {map50_score:.4f}")

    with torch.no_grad():
        total_iou = 0.0
        total_accuracy = 0.0
        total_precision = 0.0
        total_recall = 0.0
        total_map50 = 0.0
        
        for images, labels in tqdm(valid_loader):
            images = images.to(device)
            labels = labels.to(device)

            output = model(images)

            output = output.squeeze()
            output = output.detach().numpy()
            output_t = torch.from_numpy(output)
            probabilities = torch.softmax(output_t, dim=0)
            segmentation_mask = np.argmax(probabilities.numpy(), axis=0)
            segmentation_mask_tensor = torch.from_numpy(segmentation_mask).to(torch.int64)

            # Convert to one-hot encoding
            one_hot_tensor = torch.nn.functional.one_hot(segmentation_mask_tensor, num_classes=15)

            # If needed, convert to float or other dtype
            one_hot_tensor = one_hot_tensor.float()
            one_hot_tensor = one_hot_tensor.permute(2, 0, 1).unsqueeze(0)  # Shape: (1, 16, H, W)

            # Calculate metrics for each class
            batch_accuracy = []
            batch_precision = []
            batch_recall = []
            batch_map50 = []

            for class_idx in range(15):  # Assuming there are 16 classes
                pred_class_mask = one_hot_tensor[0][class_idx].cpu().numpy()
                gt_class_mask = labels[0][class_idx].cpu().numpy()

                # Calculate metrics for the current class
                acc, prec, rec = calculate_metrics(pred_class_mask, gt_class_mask)
                map50_score = calculate_map50(pred_class_mask, gt_class_mask)

                batch_accuracy.append(acc)
                batch_precision.append(prec)
                batch_recall.append(rec)
                batch_map50.append(map50_score)

            # Average metrics across all classes in the batch
            total_accuracy += np.mean(batch_accuracy)
            total_precision += np.mean(batch_precision)
            total_recall += np.mean(batch_recall)
            total_map50 += np.mean(batch_map50)

            # Calculate IoU for the batch
            total_iou += calculate_iou(one_hot_tensor.cpu().numpy(), labels.cpu().numpy())

        # Compute mean metrics across all batches
        mean_accuracy = total_accuracy / len(valid_loader)
        mean_precision = total_precision / len(valid_loader)
        mean_recall = total_recall / len(valid_loader)
        mean_map50 = total_map50 / len(valid_loader)
        mean_iou = total_iou / len(valid_loader)

        print(f"Mean IOU: {mean_iou:.4f}")
        print(f"Mean Accuracy: {mean_accuracy:.4f}")
        print(f"Mean Precision: {mean_precision:.4f}")
        print(f"Mean Recall: {mean_recall:.4f}")
        print(f"Mean mAP@50: {mean_map50:.4f}")
