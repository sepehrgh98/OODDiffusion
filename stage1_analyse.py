import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os



# Load the CSV file
file_path = os.path.join(os.getcwd(), 'results/stage1/Task1/AverageFilter+Cosine.csv')  # Replace with your CSV file path
data = pd.read_csv(file_path)

# Preview the first few rows
print(data.head())


# Extract the cosine similarity values for real and synthetic images
sim_real = data['Real_Image'].dropna().values  # Remove any NaN values if present
sim_syn = data['Synthetic_Image'].dropna().values  # Remove any NaN values if present

# Calculate statistics for real images
mean_real = np.mean(sim_real)
std_real = np.std(sim_real)

# Calculate statistics for synthetic images
mean_syn = np.mean(sim_syn)
std_syn = np.std(sim_syn)

print(f"Similarity (Real): {mean_real:.8f}, Std Dev: {std_real:.8f}")
print(f"Similarity (Synthetic): {mean_syn:.8f}, Std Dev: {std_syn:.8f}")


# Plot the distributions using Seaborn for better visualization
plt.figure(figsize=(10, 6))  # Define the figure size

# Plot histogram for synthetic images
sns.histplot(sim_syn, color='blue', label='Synthetic', kde=True, bins=30)

# Plot histogram for real images
sns.histplot(sim_real, color='red', label='Real', kde=True, bins=30)

# Set the plot labels and title
plt.xlabel('Cosine Similarity')
plt.ylabel('Frequency')
plt.title('Distribution of Average Fliter + Cosine Similarity for Synthetic and Real Images')
plt.legend()

# Show the plot
plt.show()


import numpy as np

# Combine synthetic and real values into one array for testing different thresholds
cos_sim_all = np.concatenate([sim_syn, sim_real])
labels = np.concatenate([np.ones(len(sim_syn)), np.zeros(len(sim_real))])  # 1 for Synthetic, 0 for Real

# Candidate thresholds
### metrics
# thresholds = [0.2475, 0.2477, 0.2480]
# thresholds = [183420, 183450, 183470]
# thresholds = [833.3, 833.35, 833.4]
# thresholds = [0.1045, 0.105, 0.1055]
# thresholds = [183380, 183400, 183450]

### smooth method
# thresholds = [0.2695, 0.270, 0.269]
# thresholds = [1.0001, 1.00011, 1.00012]
# thresholds = [0.2564, 0.2568, 0.2570]
thresholds = [0.2515, 0.2520, 0.2522]








# Iterate over each threshold and calculate metrics
for threshold in thresholds:
    # Predict labels based on the threshold
    predictions = (cos_sim_all >= threshold).astype(int)
    
    # Calculate TP, FP, TN, FN
    tp = np.sum((predictions == 1) & (labels == 1))
    fp = np.sum((predictions == 1) & (labels == 0))
    tn = np.sum((predictions == 0) & (labels == 0))
    fn = np.sum((predictions == 0) & (labels == 1))
    
    # Calculate classification metrics
    accuracy = (tp + tn) / len(labels)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Print metrics for the current threshold
    print(f"Threshold: {threshold}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1_score:.4f}")
    print()


