"""Inspirations:
https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-resnet34-mnist.ipynb"""

#################################################### LIBRARIES ####################################################
import os
import time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms

import matplotlib.pyplot as plt
from PIL import Image


if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True


#################################################### SETTINGS ####################################################

# Hyperparameters
RANDOM_SEED = 1
LEARNING_RATE = 0.001
BATCH_SIZE = 128
NUM_EPOCHS = 10

# Architecture
NUM_FEATURES = 28*28
NUM_CLASSES = 10

# Other
DEVICE = "mps" #cuda:2 is not mac friendly
GRAYSCALE = True
STRIDE = 2

#################################################### MNIST DATASET ####################################################
"""PLACEHOLDER ALERT: test with our own data"""

# Note transforms.ToTensor() scales input images
# to 0-1 range
train_dataset = datasets.MNIST(root='data', 
                               train=True, 
                               transform=transforms.ToTensor(),
                               download=True)

test_dataset = datasets.MNIST(root='data', 
                              train=False, 
                              transform=transforms.ToTensor())


train_loader = DataLoader(dataset=train_dataset, 
                          batch_size=BATCH_SIZE, 
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset, 
                         batch_size=BATCH_SIZE, 
                         shuffle=False)

# Checking the dataset
for images, labels in train_loader:  
    print('Image batch dimensions:', images.shape)
    print('Image label dimensions:', labels.shape)
    break

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
torch.manual_seed(0)

for epoch in range(2):

    for batch_idx, (x, y) in enumerate(train_loader):
        
        print('Epoch:', epoch+1, end='')
        print(' | Batch index:', batch_idx, end='')
        print(' | Batch size:', y.size()[0])
        
        x = x.to(device)
        y = y.to(device)
        break




#################################################### MODEL ####################################################
"""
Goal: changing the architecture of resnet a little and add our own head and neck to make it YOLO-like ;D
"""

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



class Customized(nn.Module):

    def __init__(self, block, layers, num_classes, grayscale):
        self.inplanes = 64
        if grayscale:
            in_dim = 1
        else:
            in_dim = 3
        super(Customized, self).__init__()
        
        
        #Loading the pre-trained ResNet-34
        pretrained_model = resnet34(weights=ResNet34_Weights.DEFAULT)
        self.conv1 = nn.Conv2d(in_dim, 64, kernel_size=7, stride = STRIDE, padding=3,
                               bias=False)
        
        #CHANGABLE: we can change the parameters here in the future 
        self.kept_layers = nn.Sequential(pretrained_model.bn1,
            pretrained_model.relu,
            pretrained_model.maxpool,
            )
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride =STRIDE)
        self.layer3 = self._make_layer(block, 256, layers[2], stride =STRIDE)
        self.layer4 = self._make_layer(block, 512, layers[3], stride =STRIDE)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n)**.5)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
        # Freezing layers
        for param in self.kept_layers.parameters():
            param.requires_grad = False

        #YOLO's neck and head equivalent
        #UNDER CONSTRUCTION: 🚧 👷‍♂️ numbers not matching up 
        '''self.neck = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
        )

        self.head = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 3 * (5 + num_classes), kernel_size=1)  # 3 anchor boxes
        )'''

        

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        #x = self.kept_layers(x)
        x = self.conv1(x)
        x = self.kept_layers(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        #x = self.neck(x)
        #x = self.head(x)
        # because MNIST is already 1x1 here:
        # disable avg pooling
        #x = self.avgpool(x)
        
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        #probas = F.softmax(logits, dim=1) nn.CrossEntropyLoss() applies softmax already
        return logits #, probas

#################################################### TRAINING ####################################################
'''Code to train the model :D '''

def train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs, device):
    model.to(device)
    model.train()  # Set the model to training mode
    prev_val_loss = 99.9
    val_loss = 0.0

    for epoch in range(num_epochs):
        running_loss = 0.0
        model.train()

        # Iterate through the dataloader
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            if not batch_idx % 50:
                print ('Epoch: %03d/%03d | Batch %04d/%04d | Loss: %.4f' 
                    %(epoch+1, NUM_EPOCHS, batch_idx, 
                        len(train_loader), loss))

        # Calculate average loss for the epoch
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # Check the model performance on the validation set
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            val_loss = 0.0
            print('Epoch: %03d/%03d | Accuracy: %.3f%%' % (
              epoch+1, NUM_EPOCHS, 
              compute_accuracy(model, train_loader, device=device)))
            
            for images, labels in valid_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)

            val_loss /= len(valid_loader)
            print(f"Validation Loss: {val_loss:.4f}")

        # Check for early stopping
        if val_loss > prev_val_loss:
            print("Validation loss increased. Stopping training.")
            break
        else:
            prev_val_loss = val_loss

        # Save the model after each epoch
        torch.save(model.state_dict(), f"unet_epoch_{epoch + 1}.pth")

    print("Training complete.")

def compute_accuracy(model, data_loader, device):
    correct_pred, num_examples = 0, 0
    for i, (features, targets) in enumerate(data_loader):
            
        features = features.to(device)
        targets = targets.to(device)

        logits = model(features)
        _, predicted_labels = torch.max(logits, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float()/num_examples * 100

def custom_yolo(num_classes):
    """Constructs a ResNet-34 model."""
    model = Customized(block=BasicBlock, 
                   layers=[3, 4, 6, 3],
                   num_classes=NUM_CLASSES,
                   grayscale=GRAYSCALE)
    return model


torch.manual_seed(RANDOM_SEED)
model = custom_yolo(NUM_CLASSES)


# Define the loss function and optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)  
#can use a combo of F.softmax() and F.cross_entropy() instead
criterion = nn.CrossEntropyLoss()  # Use CrossEntropyLoss for multi-class segmentation



# Transform for data augmentation
transform = transforms.Compose([
    transforms.RandomApply([transforms.ColorJitter(contrast=0.5)], p=1),
    transforms.RandomApply([transforms.ColorJitter(brightness=0.5)], p=1),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=(21, 21))], p=1),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # Normalize the image
])

train_model(model, train_loader, test_loader, criterion, optimizer, NUM_EPOCHS, device)

#################################################### TESTING ####################################################
"""PLACEHOLDER ALERT: pls replace :D """
 
with torch.set_grad_enabled(False): # save memory during inference
    print('Test accuracy: %.2f%%' % (compute_accuracy(model, test_loader, device=DEVICE)))

for batch_idx, (features, targets) in enumerate(test_loader):
    features = features
    targets = targets
    break
    
    
nhwc_img = np.transpose(features[0], axes=(1, 2, 0))
nhw_img = np.squeeze(nhwc_img.numpy(), axis=2)
plt.imshow(nhw_img, cmap='Greys')


model.eval()
logits = model(features.to(device)[0, None])
probas = F.softmax(logits, dim=1)
print('Probability 7 %.2f%%' % (probas[0][7]*100))


