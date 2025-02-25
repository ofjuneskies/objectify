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

valid_images_folder = 'C:/Users/jenny/OneDrive/Projects/UAV/senior-design-proj/Unet/yolo/images/validation'
valid_labels_folder = 'C:/Users/jenny/OneDrive/Projects/UAV/senior-design-proj/Unet/yolo/labels/validation'
valid_dataset = ForgeDataset(images_folder=valid_images_folder, labels_folder=valid_labels_folder, transform=transform)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

state_dict = torch.load('C:/Users/jenny/OneDrive/Projects/UAV/senior-design-proj/Unet/unet_v5.pth', map_location=torch.device(device))
model = UNet(num_classes=16).to(device)
model.load_state_dict(state_dict)

model.eval()

def calculate_iou(pred_mask, gt_mask):
    """
    Calculate Intersection over Union (IoU) for a single pair of predicted and ground truth masks.

    Args:
        pred_mask (np.ndarray): Predicted segmentation mask.
        gt_mask (np.ndarray): Ground truth segmentation mask.

    Returns:
        float: IoU score.
    """
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    iou = intersection / union if union != 0 else 0
    return iou
    


if __name__ == "__main__":
    with torch.no_grad():
        iou = 0.0
        for images, labels in tqdm(valid_loader):
            images = images.to(device)
            labels = labels.to(device)

            output = model(images)
            # for i in range(16):
                # preds = torch.softmax(output[i, :, :], dim=0)

                # preds = preds.long()
                # preds = F.one_hot(preds, 1)

                # target = labels[i, :, :].long()
                # target = F.one_hot(target, 1)

                # miou = MeanIoU(num_classes=1).to(device)
                # iou += miou(preds, target)

            iou += calculate_iou(output.cpu().numpy(), labels.cpu().numpy())

        meanMIOU = iou / len(valid_loader)
        print(f"Mean MIOU: {meanMIOU:.4f}")


    # preds = randint(1, 2, (10, 3, 128, 128))
    # target = randint(1, 2, (10, 3, 128, 128))
    # print(preds)
    # miou = MeanIoU(num_classes=16)
    # print(miou(preds, target))




