import os
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image, ImageDraw

WIDTH, HEIGHT, NUM_CLASSES = 640, 640, 15

def get_polygon(file_path):
    with open(file_path) as file:
        data = file.read().strip().split("\n")
    polygons = []
    for line in data:
        values = list(map(float, line.split()))
        if len(values) > 0:
            class_id = int(values[0])
            coords = [(x * WIDTH, y * HEIGHT) for x, y in zip(values[1::2], values[2::2])]
            polygons.append((class_id, coords))
    return polygons

def conv_polygon_to_tensor(coords):
    mask = Image.new('L', (HEIGHT, WIDTH), 0)
    draw = ImageDraw.Draw(mask)
    draw.polygon(coords, outline=1, fill=1)
    mask_np = np.array(mask, dtype=int)
    return torch.from_numpy(mask_np)

def get_img_label_tensor(id, image_path, label_path, transform=None):
    # Get the image tensor
    image = Image.open(image_path).convert("RGB")
    if transform is None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    img_tensor = transform(image)
    
    # Get the label tensor
    polygons = get_polygon(label_path)
    label_tensor = torch.zeros(WIDTH, HEIGHT, NUM_CLASSES + 1)
    background_tensor = torch.zeros(WIDTH, HEIGHT)
    for polygon in polygons:
        label_tensor[:, :, polygon[0]] = conv_polygon_to_tensor(polygon[1])
    label_tensor = torch.transpose(label_tensor, 0, 2)
    label_tensor = torch.transpose(label_tensor, 1, 2)

    for class_id in range(NUM_CLASSES):
        background_tensor = torch.logical_or(background_tensor, label_tensor[class_id, :, :])   
    label_tensor[NUM_CLASSES, :, :] = (~background_tensor).to(dtype=torch.float)
    return (img_tensor, label_tensor)

if __name__ == "__main__":
    tup = get_img_label_tensor(0, './dataset/images/train/image1.png', './dataset/labels/train/image1.txt')
    print(tup)
    print(tup[0].shape)
    print(tup[1].shape)
    print(tup[1][15, :, :].shape)
    #torch.save(tup[1][15, :, :], "tensor.txt")
    np.savetxt("binary_tensor.txt", tup[1][15, :, :].numpy(), fmt="%d")

