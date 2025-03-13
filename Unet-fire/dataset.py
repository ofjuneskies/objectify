import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader

class ForgeDataset(Dataset):
    def __init__(self, images_folder, labels_folder, transform=None):
        self.images_folder = images_folder
        self.labels_folder = labels_folder
        self.transform = transform
        
        # List all images and labels
        self.image_files = sorted([os.path.join(root, filename) 
                                   for root, _, files in os.walk(images_folder) 
                                   for filename in files if filename.endswith('.jpg')])
        
        self.label_files = sorted([os.path.join(root, filename) 
                                   for root, _, files in os.walk(labels_folder) 
                                   for filename in files if filename.endswith('.png')])
        
        # Sanity check to ensure the dataset is matched properly
        assert len(self.image_files) == len(self.label_files), f"Number of images and labels do not match."

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        label_path = self.label_files[idx]
        image = Image.open(image_path).convert("RGB")
        label = Image.open(label_path).convert("L")
        if self.transform is None:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        img_tensor = transform(image)

        label_tensor = torch.from_numpy(np.array(label)).long()
        one_hot_label = torch.zeros(2, label_tensor.size(0), label_tensor.size(1))
        one_hot_label[0] = (label_tensor == 0).float()  # Class 0
        one_hot_label[1] = (label_tensor == 1).float()  # Class 1

        return img_tensor, one_hot_label

if __name__ == "__main__":
    # Create the dataset and dataloader
    train_images_folder = 'wildfire/train'
    train_labels_folder = 'wildfire/train'
    train_dataset = ForgeDataset(images_folder=train_images_folder, labels_folder=train_labels_folder, transform=None)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)

    valid_images_folder = 'wildfire/valid'
    valid_labels_folder = 'wildfire/valid'
    valid_dataset = ForgeDataset(images_folder=valid_images_folder, labels_folder=valid_labels_folder, transform=None)
    valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False, num_workers=4)

    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(valid_dataset)}")

    image, label = train_dataset[0]
    print(f"Image shape: {image.shape}")
    print(f"Label shape: {label.shape}")
    print(f"Label unique values: {torch.unique(label)}")