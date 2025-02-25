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