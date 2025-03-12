import matplotlib.pyplot as plt
import numpy as np

# Data for the bar graph
classes = ['mannequin', 'suitcase', 'tennisracket', 'boat', 'stopsign', 'plane', 'baseballbat', 'bus', 'mattress', 'skis', 'umbrella', 'snowboard', 'motorcycle', 'car', 'sportsball']

unet_v3 = [0.0250, 0.4161, 0.5882, 0.8145, 0.6781, 0.6751, 0.3056, 0.9315, 0.6476, 0.8555, 0.9734, 0.6678, 0.8826, 0.8412, 0.4409]
unet_v6 = [0.5709, 0.5631, 0.8201, 0.8702, 0.8118, 0.8766, 0.5733, 0.9583, 0.8266, 0.9322, 0.9923, 0.9125, 0.9283, 0.9581, 0.6369]
unet_v7 = [0.5384, 0.6081, 0.7064, 0.8688, 0.7808, 0.8426, 0.5240, 0.9485, 0.8085, 0.9400, 0.9870, 0.8786, 0.9323, 0.9320, 0.5627]
unet_v8 = [0.6218, 0.5800, 0.7024, 0.8520, 0.8065, 0.8671, 0.5739, 0.9662, 0.8237, 0.9554, 0.9916, 0.8720, 0.9449, 0.9619, 0.6712]
yolo_v11 = [0.9134, 0.9649, 0.9169, 0.9341, 0.8302, 0.7853, 0.6704, 0.9070, 0.9528, 0.9788, 0.9883, 0.9713, 0.9685, 0.9830, 0.9114]

x = np.arange(len(classes))  # the label locations
width = 0.15  # the width of the bars

fig, ax = plt.subplots(figsize=(15, 8))
rects1 = ax.bar(x - 2*width, unet_v3, width, label='Unet v3')
rects2 = ax.bar(x - width, unet_v6, width, label='Unet v6')
rects3 = ax.bar(x, unet_v7, width, label='Unet v7')
rects4 = ax.bar(x + width, unet_v8, width, label='Unet v8')
rects5 = ax.bar(x + 2*width, yolo_v11, width, label='YOLO v11')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Classes')
ax.set_ylabel('Accuracy')
ax.set_title('Per-class Accuracy for Different Models')
ax.set_xticks(x)
ax.set_xticklabels(classes, rotation=45, ha='right')
ax.legend()

fig.tight_layout()
plt.savefig('per_class_accuracy.png')

plt.show()