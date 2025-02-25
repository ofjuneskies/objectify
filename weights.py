import os
from collections import Counter

folder_path = 'dataset/labels/train'

class_counter = Counter()

for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        file_path = os.path.join(folder_path, filename)
        # Read the file and count class occurrences
        with open(file_path, 'r') as file:
            for line in file:
                class_id = line.split()[0]  # Get the first number (class) of each line
                class_counter[class_id] += 1

# Convert the counter to a dictionary for easier use
class_frequencies = dict(class_counter)

# Print the class frequencies
print("Class Frequencies:", class_frequencies)

# Calculate total pixels
total_pixels = sum(class_frequencies.values())

# Compute weights inversely proportional to frequencies
weights = {class_id: total_pixels / freq for class_id, freq in class_frequencies.items()}

# Normalize weights by dividing by the maximum weight
max_weight = max(weights.values())
normalized_weights = {class_id: weight / max_weight for class_id, weight in weights.items()}

# Print the normalized weights
sorted_dict = dict(sorted(normalized_weights.items(), key=lambda item: int(item[0])))
# print values to 3 decimal places
print("Normalized Weights:", [round(v, 3) for k, v in sorted_dict.items()])

print("Min Weight:", min(normalized_weights.values()))

import matplotlib.pyplot as plt

# Extract keys and values
weights = dict(sorted(class_frequencies.items(), key=lambda item: int(item[0])))
class_names = {
    0: "mannequin",
    1: "suitcase",
    2: "tennisracket",
    3: "boat",
    4: "stopsign",
    5: "plane",
    6: "baseballbat",
    7: "bus",
    8: "mattress",
    9: "skis",
    10: "umbrella",
    11: "snowboard",
    12: "motorcycle",
    13: "car",
    14: "sportsball"
}
keys = list(class_names.values())
values = list(weights.values())

# Create the bar graph
plt.figure(figsize=(6, 6))
plt.bar(keys, values)
plt.xlabel('Classes')
plt.ylabel('Instances')
plt.xticks(rotation=90)
plt.title('Class Frequency')
plt.tight_layout()
plt.show()