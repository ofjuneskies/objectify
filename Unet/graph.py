import matplotlib.pyplot as plt
import numpy as np

# Data preparation
models = ['YOLO v11', 'Unet v1', 'Unet v2', 'Unet v3', 'Unet v4', 'Unet v5', 'Unet v6', 'Unet v7', 'Unet v8']
val_loss = [1.22134, 0.0303, 0.0338, 0.1006, 0.0258, 0.0648, 0.0251, 0.0301, 0.0309]
mean_iou = [0.0073, 0.0691, 0.0687, 0.9515, 0.9150, 0.6957, 0.9170, 0.9314, 0.8971]
mean_acc = [0.9892, 0.9376, 0.9375, 0.9980, 0.9965, 0.9861, 0.9966, 0.9971,  0.9958]
mean_prec = [0.7077, 0.2030, 0.1784, 0.8448, 0.7196, 0.4402, 0.7372, 0.7703, 0.6873]
mean_recall = [0.7136, 0.3321, 0.2999, 0.8164, 0.7670, 0.4195, 0.7820, 0.7965, 0.7463]
mean_map50 = [0.7059, 0.1779, 0.1543, 0.8121, 0.7205, 0.3948, 0.7338, 0.7636, 0.6776]

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

# Make the plot# A more appealing color palette
plt.bar(r1, val_loss, width=barWidth, label='Validation Loss', color='#4e79a7')
plt.bar(r2, mean_iou, width=barWidth, label='Mean IOU', color='#f28e2b')
plt.bar(r3, mean_acc, width=barWidth, label='Mean Accuracy', color='#59a14f')
plt.bar(r4, mean_prec, width=barWidth, label='Mean Precision', color='#e15759')
plt.bar(r5, mean_recall, width=barWidth, label='Mean Recall', color='#76b7b2')
plt.bar(r6, mean_map50, width=barWidth, label='Mean mAP@50', color='#b07aa1')

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
plt.savefig('unet_performance_comparison.png')
plt.show()
