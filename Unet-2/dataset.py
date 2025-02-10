import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from process import get_img_label_tensor

class ForgeDataset(Dataset):
    def __init__(self, images_folder, labels_folder, transform=None):
        self.images_folder = images_folder
        self.labels_folder = labels_folder
        self.transform = transform
        
        # List all images and labels
        self.image_files = sorted([os.path.join(root, filename) 
                                   for root, _, files in os.walk(images_folder) 
                                   for filename in files if filename.endswith('.png')])
        
        self.label_files = sorted([os.path.join(root, filename) 
                                   for root, _, files in os.walk(labels_folder) 
                                   for filename in files if filename.endswith('.txt')])
        
        # Sanity check to ensure the dataset is matched properly
        assert len(self.image_files) == len(self.label_files), "Number of images and labels do not match."

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        label_path = self.label_files[idx]
        try:
            image, label = get_img_label_tensor(idx, image_path, label_path, self.transform)
            return image, label
        except FileNotFoundError as e:
            image = torch.zeros(3, 640, 640)
            label = torch.zeros(15, 640, 640)
            return image, label