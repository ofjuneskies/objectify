import matplotlib.pyplot as plt
import numpy as np

# Data preparation
models = ['Unet v1', 'Unet v2', 'Unet v3', 'Unet v4', 'Unet v5', 'Unet v6']
val_loss = [0.0303, 0.0338, 0.1006, 0.0258, 0.0648, 0.0251]
mean_iou = [0.0691, 0.0687, 0.9515, 0.9150, 0.6957, 0.9170]
mean_acc = [0.9376, 0.9375, 0.9967, 0.9941, 0.9762, 0.9942]
mean_prec = [0.2030, 0.1784, 0.8539, 0.7370, 0.4747, 0.7535]
mean_recall = [0.3321, 0.2999, 0.8269, 0.7785, 0.4444, 0.7926]
mean_map50 = [0.1779, 0.1543, 0.8239, 0.7379, 0.4319, 0.7504]

# Set width of bars
barWidth = 0.12
 
# Set positions of the bars on X axis
r1 = np.arange(len(models))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]
r5 = [x + barWidth for x in r4]
r6 = [x + barWidth for x in r5]

# Create the figure and axes
plt.figure(figsize=(14, 8))

# Make the plot
plt.bar(r1, val_loss, width=barWidth, label='Validation Loss', color='#1f77b4')
plt.bar(r2, mean_iou, width=barWidth, label='Mean IOU', color='#ff7f0e')
plt.bar(r3, mean_acc, width=barWidth, label='Mean Accuracy', color='#2ca02c')
plt.bar(r4, mean_prec, width=barWidth, label='Mean Precision', color='#d62728')
plt.bar(r5, mean_recall, width=barWidth, label='Mean Recall', color='#9467bd')
plt.bar(r6, mean_map50, width=barWidth, label='Mean mAP@50', color='#8c564b')

# Add labels and title
plt.xlabel('Model Versions', fontweight='bold', fontsize=12)
plt.ylabel('Scores', fontweight='bold', fontsize=12)
plt.title('Performance Comparison of Unet Versions', fontweight='bold', fontsize=14)

# Add xticks on the middle of the group bars
plt.xticks([r + barWidth*2.5 for r in range(len(models))], models)

# Create legend & Show graphic
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.07), ncol=6)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add a horizontal line at y=1.0 to show the maximum possible value for metrics
plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

# Adjust layout and display the plot
plt.tight_layout()
plt.show()
