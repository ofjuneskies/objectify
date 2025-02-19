import os
import time
import torch.nn as nn
from torchvision.models import resnet34

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms

import matplotlib.pyplot as plt
from PIL import Image
import cv2



output = np.load("/Users/Thao/Downloads/output.npy")
image = Image.open("/Users/Thao/Downloads/499E0285-49A0-4F41-B2C9-BF50B68FE7D4.png").convert("RGB")
output = output.squeeze() # numpy.ndarray

output_t =  torch.from_numpy(output) # converts it to tensor
probabilities = torch.softmax(output_t, dim=0)
probabilities_np = probabilities.numpy() # converts back to numPy
segmentation_mask = np.argmax(probabilities_np, axis = 0) #suppose to get the class with the highest probability
segmentation_mask = segmentation_mask.astype(np.uint8) # convert to integer type
segmentation_mask= cv2.resize(segmentation_mask, image.size, interpolation=cv2.INTER_NEAREST)

plt.figure(figsize=(10, 8))
plt.imshow(image)
plt.imshow(segmentation_mask, cmap="tab20", alpha=0.5, interpolation = "nearest")  # 'jet' for colorful segmentation
plt.colorbar()
plt.axis("off")

#code to see each channels
#fig, axes = plt.subplots(1,1,figsize = (15,9))
#for i in range(15):
#    axes[i].imshow(probabilities[i], cmap="jet")
#    axes[i].set_title(f'Channel {i}')
#    axes[i].axis("off")

#plt.tight_layout()

plt.show()



start_time = time.time()
for epoch in range(NUM_EPOCHS):
    
    model.train()
    for batch_idx, (features, targets) in enumerate(train_loader):
        
        features = features.to(DEVICE)
        targets = targets.to(DEVICE)
            
        ### FORWARD AND BACK PROP
        logits, probas = model(features)
        cost = F.cross_entropy(logits, targets)
        optimizer.zero_grad()
        
        cost.backward()
        
        ### UPDATE MODEL PARAMETERS
        #optimizer.step()
        
        ### LOGGING
        if not batch_idx % 50:
            print ('Epoch: %03d/%03d | Batch %04d/%04d | Cost: %.4f' 
                   %(epoch+1, NUM_EPOCHS, batch_idx, 
                     len(train_loader), cost))

        

    model.eval()
    with torch.set_grad_enabled(False): # save memory during inference
        print('Epoch: %03d/%03d | Train: %.3f%%' % (
              epoch+1, NUM_EPOCHS, 
              compute_accuracy(model, train_loader, device=DEVICE)))
        
    print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
    
print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))



