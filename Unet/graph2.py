import matplotlib.pyplot as plt
import numpy as np

# Data for UNet v3, v6, v7, v8, and YOLO v11-100
models = ['UNet v3', 'UNet v6', 'UNet v7', 'UNet v8', 'YOLO v11-100']

metrics = ['Mean IOU', 'Mean Accuracy', 'Mean Precision', 'Mean Recall', 'Mean mAP@50']

# Values from the results
values = np.array([
    # UNet v3
    [0.9515, 0.9980, 0.8448, 0.8164, 0.8121],
    # UNet v6
    [0.9170, 0.9966, 0.7372, 0.7820, 0.7338],
    # UNet v7
    [0.9314, 0.9971, 0.7703, 0.7965, 0.7636],
    # UNet v8
    [0.8971, 0.9958, 0.6873, 0.7463, 0.6776],
    # YOLO v11-100
    [0.0073, 0.9892, 0.7077, 0.7136, 0.7059]
])

# Create a figure with 2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Bar chart for overall metrics
x = np.arange(len(metrics))
width = 0.15  # Reduced width to accommodate 5 models

for i, model in enumerate(models):
    ax1.bar(x + i*width, values[i], width, label=model)

ax1.set_ylabel('Score')
ax1.set_title('Overall Performance Metrics Comparison')
ax1.set_xticks(x + width*2)  # Adjusted to center the labels
ax1.set_xticklabels(metrics)
ax1.legend()
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# Radar chart for the same data
angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]  # Close the loop

# Add the values for each model, and close the loop for plotting
values_radar = np.zeros((len(models), len(metrics)+1))
for i in range(len(models)):
    values_radar[i, :-1] = values[i]
    values_radar[i, -1] = values[i][0]  # Close the loop

# Plot radar chart
ax2 = plt.subplot(122, polar=True)
ax2.set_theta_offset(np.pi / 2)
ax2.set_theta_direction(-1)
ax2.set_thetagrids(np.degrees(angles[:-1]), metrics)

for i, model in enumerate(models):
    ax2.plot(angles, values_radar[i], linewidth=2, label=model)
    ax2.fill(angles, values_radar[i], alpha=0.1)

ax2.set_title('Radar Plot of Performance Metrics')
ax2.legend(loc='upper right')

plt.tight_layout()
plt.savefig('unet_metrics_comparison.png')
plt.show()

# Plot class-wise performance for each model
fig, axes = plt.subplots(3, 1, figsize=(15, 18))

# Class numbers
classes = list(range(15))  # 0-14

# Metrics to plot for each class
class_metrics = ['Mean Precision', 'Mean Recall', 'Mean mAP@50']

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# Data for each model
# UNet v3
unet3_precision = [0.7721, 0.8037, 0.8313, 0.8483, 0.9141, 0.8657, 0.8753, 0.7838, 0.7858, 0.8380, 0.9488, 0.8522, 0.8929, 0.8886, 0.7720]
unet3_recall = [0.7537, 0.7808, 0.8020, 0.8269, 0.8824, 0.8446, 0.8217, 0.7925, 0.7325, 0.8185, 0.9500, 0.8300, 0.8813, 0.8645, 0.6642]
unet3_map = [0.7500, 0.7893, 0.8214, 0.8214, 0.8964, 0.8250, 0.8071, 0.7929, 0.7143, 0.8286, 0.9500, 0.8357, 0.8857, 0.8571, 0.6071]

# UNet v6
unet6_precision = [0.6056, 0.6353, 0.5827, 0.7774, 0.7221, 0.7956, 0.7338, 0.7609, 0.6465, 0.7545, 0.9286, 0.7777, 0.8879, 0.7809, 0.6689]
unet6_recall = [0.6638, 0.6727, 0.6402, 0.7892, 0.7784, 0.8489, 0.7979, 0.7924, 0.7084, 0.7957, 0.9577, 0.8170, 0.9276, 0.8246, 0.7146]
unet6_map = [0.5607, 0.6250, 0.5714, 0.7821, 0.7143, 0.7750, 0.6893, 0.7821, 0.6464, 0.7821, 0.9500, 0.7964, 0.9107, 0.7893, 0.6321]

# UNet v7
unet7_precision = [0.6384, 0.6600, 0.7664, 0.7630, 0.7647, 0.7724, 0.7683, 0.7565, 0.6946, 0.8209, 0.9398, 0.7885, 0.9069, 0.7852, 0.7295]
unet7_recall = [0.6706, 0.6845, 0.7925, 0.7652, 0.7980, 0.8092, 0.8033, 0.7800, 0.7385, 0.8471, 0.9634, 0.8096, 0.9397, 0.8092, 0.7363]
unet7_map = [0.6143, 0.6536, 0.7750, 0.7607, 0.7536, 0.7607, 0.7179, 0.7714, 0.6750, 0.8393, 0.9607, 0.8107, 0.9286, 0.7750, 0.6571]

# UNet v8
unet8_precision = [0.5241, 0.5608, 0.5974, 0.6660, 0.6799, 0.7661, 0.6996, 0.6752, 0.5033, 0.8229, 0.9118, 0.7180, 0.8508, 0.7406, 0.5929]
unet8_recall = [0.5989, 0.6045, 0.6559, 0.6813, 0.7490, 0.8258, 0.7721, 0.7153, 0.5857, 0.8730, 0.9533, 0.7629, 0.9036, 0.7980, 0.7145]
unet8_map = [0.4821, 0.5464, 0.5571, 0.6750, 0.6393, 0.7429, 0.6643, 0.7036, 0.4929, 0.8393, 0.9464, 0.6929, 0.8893, 0.7607, 0.5321]

# YOLO v11-100
yolo_precision = [0, 0.8087, 0.7689, 0.8067, 0.4616, 0.6218, 0.8046, 0.6498, 0.6729, 0.8036, 0.8526, 0.8606, 0.8107, 0.6729, 0.3121]
yolo_recall = [0, 0.8116, 0.7898, 0.8070, 0.4687, 0.6252, 0.8085, 0.6512, 0.6798, 0.8036, 0.8601, 0.8564, 0.8107, 0.6773, 0.3403]
yolo_map = [0, 0.8107, 0.7750, 0.8071, 0.4607, 0.6071, 0.8071, 0.6429, 0.6714, 0.8036, 0.8536, 0.8571, 0.8107, 0.6679, 0.3071]

# Plot for Precision
axes[0].set_title('Precision per Class')
axes[0].bar(np.array(classes) - 0.3, unet3_precision, width=0.15, color=colors[0], label='UNet v3')
axes[0].bar(np.array(classes) - 0.15, unet6_precision, width=0.15, color=colors[1], label='UNet v6')
axes[0].bar(np.array(classes), unet7_precision, width=0.15, color=colors[2], label='UNet v7')
axes[0].bar(np.array(classes) + 0.15, unet8_precision, width=0.15, color=colors[3], label='UNet v8')
axes[0].bar(np.array(classes) + 0.3, yolo_precision, width=0.15, color=colors[4], label='YOLO v11-100')
axes[0].set_xticks(classes)
axes[0].set_xlabel('Class')
axes[0].set_ylabel('Score')
axes[0].legend()
axes[0].grid(axis='y', linestyle='--', alpha=0.7)

# Plot for Recall
axes[1].set_title('Recall per Class')
axes[1].bar(np.array(classes) - 0.3, unet3_recall, width=0.15, color=colors[0], label='UNet v3')
axes[1].bar(np.array(classes) - 0.15, unet6_recall, width=0.15, color=colors[1], label='UNet v6')
axes[1].bar(np.array(classes), unet7_recall, width=0.15, color=colors[2], label='UNet v7')
axes[1].bar(np.array(classes) + 0.15, unet8_recall, width=0.15, color=colors[3], label='UNet v8')
axes[1].bar(np.array(classes) + 0.3, yolo_recall, width=0.15, color=colors[4], label='YOLO v11-100')
axes[1].set_xticks(classes)
axes[1].set_xlabel('Class')
axes[1].set_ylabel('Score')
axes[1].legend()
axes[1].grid(axis='y', linestyle='--', alpha=0.7)

# Plot for mAP@50
axes[2].set_title('mAP@50 per Class')
axes[2].bar(np.array(classes) - 0.3, unet3_map, width=0.15, color=colors[0], label='UNet v3')
axes[2].bar(np.array(classes) - 0.15, unet6_map, width=0.15, color=colors[1], label='UNet v6')
axes[2].bar(np.array(classes), unet7_map, width=0.15, color=colors[2], label='UNet v7')
axes[2].bar(np.array(classes) + 0.15, unet8_map, width=0.15, color=colors[3], label='UNet v8')
axes[2].bar(np.array(classes) + 0.3, yolo_map, width=0.15, color=colors[4], label='YOLO v11-100')
axes[2].set_xticks(classes)
axes[2].set_xlabel('Class')
axes[2].set_ylabel('Score')
axes[2].legend()
axes[2].grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('unet_metrics_per_class.png')
plt.show()
