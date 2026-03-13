# SNN Data Preprocessing - Quick Reference Guide

## 🎯 What is Data Preprocessing?

Data preprocessing is the crucial step of cleaning and transforming your raw spike train data before training your SNN model. It improves model performance and ensures data quality.

## 📋 Why Preprocess SNN Datasets?

### Common Issues in Spike Train Data:
1. **Noisy neurons** - Some neurons fire too much or too little
2. **Silent samples** - Some samples have almost no spikes
3. **Imbalanced classes** - Unequal number of samples per class
4. **Irregular firing rates** - Inconsistent spike patterns
5. **Limited data** - Need augmentation for better generalization

## 🚀 Quick Start (One Command)

```python
from snn_data_preprocessing import SNNDataPreprocessor

# Initialize
preprocessor = SNNDataPreprocessor(random_state=42)

# Run complete pipeline
result = preprocessor.preprocess_pipeline(
    dataset_name='classification_rate',
    remove_noise=True,      # Remove bad neurons/samples
    normalize=True,         # Normalize firing rates
    balance=False,          # Balance classes (if imbalanced)
    augment=False,          # Create augmented copies
    split=True,             # Split train/val/test
    save_processed=True     # Save to disk
)

# Access processed data
train_data = result['splits']['train']
train_spikes = train_data['spike_trains']  # Ready to use!
train_labels = train_data['labels']
```

## 🔧 Preprocessing Operations

### 1. Noise Removal

**Remove Noisy Neurons:**
```python
# Removes neurons with abnormal firing rates
filtered_trains, kept_neurons = preprocessor.remove_noisy_neurons(
    spike_trains,
    threshold=0.01  # Remove if <1% or >99% firing rate
)
```

**Remove Silent Samples:**
```python
# Removes samples with too few spikes
cleaned_trains, cleaned_labels = preprocessor.remove_silent_samples(
    spike_trains,
    labels,
    min_spikes=10  # Minimum spikes required
)
```

### 2. Normalization

**Normalize Firing Rates:**
```python
# Three methods available
normalized = preprocessor.normalize_spike_rates(
    spike_trains,
    method='global'  # 'global', 'per_neuron', or 'per_sample'
)
```

- **global**: Normalize all data to target firing rate
- **per_neuron**: Each neuron independently normalized
- **per_sample**: Each sample independently normalized

### 3. Class Balancing

**Balance Class Distribution:**
```python
# Two methods
balanced_trains, balanced_labels = preprocessor.balance_classes(
    spike_trains,
    labels,
    method='undersample'  # 'undersample' or 'oversample'
)
```

- **undersample**: Reduce to smallest class size
- **oversample**: Increase to largest class size (with replacement)

### 4. Data Augmentation

**Temporal Jitter:**
```python
# Shift spike times slightly
jittered = preprocessor.augment_jitter(
    spike_trains,
    jitter_std=2.0  # Standard deviation in time steps
)
```

**Spike Dropout:**
```python
# Randomly drop spikes
dropout = preprocessor.augment_dropout(
    spike_trains,
    dropout_rate=0.1  # 10% of spikes dropped
)
```

**Add Noise:**
```python
# Add random noise spikes
noisy = preprocessor.augment_noise(
    spike_trains,
    noise_rate=0.01  # 1% noise
)
```

**Time Warp:**
```python
# Stretch/compress time dimension
warped = preprocessor.augment_time_warp(
    spike_trains,
    warp_factor=0.1  # ±10% time warping
)
```

### 5. Train/Val/Test Split

**Split Dataset:**
```python
splits = preprocessor.split_dataset(
    spike_trains,
    labels,
    train_ratio=0.7,    # 70% training
    val_ratio=0.15,     # 15% validation
    test_ratio=0.15,    # 15% testing
    stratify=True       # Maintain class distribution
)

# Access splits
train_spikes = splits['train']['spike_trains']
train_labels = splits['train']['labels']
val_spikes = splits['val']['spike_trains']
test_spikes = splits['test']['spike_trains']
```

### 6. Data Quality Analysis

**Analyze Quality:**
```python
quality_stats = preprocessor.analyze_data_quality(
    spike_trains,
    labels,
    save_path='quality_report.png'  # Save visualization
)

# Returns dictionary with statistics
print(quality_stats['total_spikes'])
print(quality_stats['sparsity'])
print(quality_stats['mean_firing_rate'])
```

## 📊 Typical Preprocessing Pipeline

### For Classification Tasks:

```python
# 1. Load data
dataset = preprocessor.load_dataset('classification_rate')
spike_trains = dataset['spike_trains']
labels = dataset['labels']

# 2. Remove noise
spike_trains, _ = preprocessor.remove_noisy_neurons(spike_trains)
spike_trains, labels = preprocessor.remove_silent_samples(spike_trains, labels)

# 3. Normalize
spike_trains = preprocessor.normalize_spike_rates(spike_trains, method='global')

# 4. Split
splits = preprocessor.split_dataset(spike_trains, labels)

# 5. Ready for training!
train_data = splits['train']
```

### For Clustering Tasks:

```python
# Similar but no class balancing needed
dataset = preprocessor.load_dataset('clustering_rate')
spike_trains = dataset['spike_trains']
labels = dataset['labels']  # Ground truth for evaluation only

# Clean and normalize
spike_trains, _ = preprocessor.remove_noisy_neurons(spike_trains)
spike_trains = preprocessor.normalize_spike_rates(spike_trains)

# Split
splits = preprocessor.split_dataset(spike_trains, labels)
```

### For Small Datasets (Need Augmentation):

```python
# Load and clean
dataset = preprocessor.load_dataset('temporal_patterns')
spike_trains = dataset['spike_trains']
labels = dataset['labels']

# Augment to increase size
augmented_list = [spike_trains]
labels_list = [labels]

# Create 3 augmented versions
for _ in range(3):
    aug = preprocessor.augment_jitter(spike_trains, jitter_std=2.0)
    augmented_list.append(aug)
    labels_list.append(labels)

# Combine
all_spikes = np.vstack(augmented_list)
all_labels = np.concatenate(labels_list)

print(f"Original: {len(labels)}, Augmented: {len(all_labels)}")

# Now split
splits = preprocessor.split_dataset(all_spikes, all_labels)
```

## 💾 Saving and Loading

### Save Preprocessed Data:

```python
# After preprocessing
np.save('preprocessed_train.npy', train_spikes)
np.save('preprocessed_train_labels.npy', train_labels)
np.save('preprocessed_test.npy', test_spikes)
np.save('preprocessed_test_labels.npy', test_labels)
```

### Load Preprocessed Data:

```python
# For training
train_spikes = np.load('preprocessed_train.npy')
train_labels = np.load('preprocessed_train_labels.npy')
test_spikes = np.load('preprocessed_test.npy')
test_labels = np.load('preprocessed_test_labels.npy')

print(f"Loaded {len(train_labels)} training samples")
print(f"Shape: {train_spikes.shape}")
```

## 🎓 Best Practices

### 1. Always Remove Noise First
```python
# Clean data before other operations
spike_trains, _ = preprocessor.remove_noisy_neurons(spike_trains)
spike_trains, labels = preprocessor.remove_silent_samples(spike_trains, labels)
```

### 2. Normalize Before Training
```python
# Helps model convergence
spike_trains = preprocessor.normalize_spike_rates(spike_trains, method='global')
```

### 3. Use Stratified Splitting
```python
# Maintains class distribution
splits = preprocessor.split_dataset(spike_trains, labels, stratify=True)
```

### 4. Analyze Quality
```python
# Always check data quality
quality_stats = preprocessor.analyze_data_quality(train_spikes, train_labels)
```

### 5. Save Preprocessing Stats
```python
# Document what you did
import json
stats = {
    'removed_neurons': 5,
    'removed_samples': 12,
    'normalization': 'global',
    'augmentation': 'jitter + dropout',
    'split_ratio': '70/15/15'
}
with open('preprocessing_info.json', 'w') as f:
    json.dump(stats, f, indent=2)
```

## ⚠️ Common Mistakes to Avoid

### ❌ DON'T:
```python
# DON'T normalize after splitting (data leakage!)
splits = preprocessor.split_dataset(spike_trains, labels)
train_normalized = preprocessor.normalize_spike_rates(splits['train']['spike_trains'])

# DON'T augment test data
test_augmented = preprocessor.augment_jitter(test_spikes)  # Wrong!

# DON'T remove noise separately per split
train_cleaned, _ = preprocessor.remove_noisy_neurons(splits['train']['spike_trains'])
```

### ✅ DO:
```python
# DO normalize before splitting
spike_trains = preprocessor.normalize_spike_rates(spike_trains)
splits = preprocessor.split_dataset(spike_trains, labels)

# DO only augment training data
train_augmented = preprocessor.augment_jitter(splits['train']['spike_trains'])

# DO remove noise on entire dataset first
spike_trains, _ = preprocessor.remove_noisy_neurons(spike_trains)
splits = preprocessor.split_dataset(spike_trains, labels)
```

## 📈 Expected Results

### Before Preprocessing:
```
Shape: (1000, 20, 100)
Total spikes: 100,523
Sparsity: 5.03%
Mean spikes/sample: 100.52 ± 45.32
Noisy neurons: 3
Silent samples: 8
```

### After Preprocessing:
```
Shape: (992, 17, 100)  # Removed 3 neurons, 8 samples
Total spikes: 95,120
Sparsity: 5.64%
Mean spikes/sample: 95.89 ± 12.15  # More consistent!
Noisy neurons: 0
Silent samples: 0
```

## 🔍 Debugging Tips

### Check Data Shape:
```python
print(f"Spike trains shape: {spike_trains.shape}")
# Should be: (n_samples, n_neurons, time_steps)
```

### Check Spike Values:
```python
print(f"Unique values: {np.unique(spike_trains)}")
# Should be: [0 1] only
```

### Check Sparsity:
```python
sparsity = (spike_trains.sum() / spike_trains.size) * 100
print(f"Sparsity: {sparsity:.2f}%")
# Typical: 5-10% for rate encoding, 1-3% for temporal
```

### Visualize Sample:
```python
import matplotlib.pyplot as plt

sample = spike_trains[0]
spike_times, neurons = np.where(sample.T)
plt.scatter(spike_times, neurons, s=5, marker='|')
plt.xlabel('Time')
plt.ylabel('Neuron')
plt.title(f'Sample 0 (Total spikes: {sample.sum()})')
plt.show()
```

## 📚 For Your Research

### What to Report in Your Paper:

1. **Preprocessing Steps:**
   - "Data was preprocessed by removing neurons with firing rates <1%, normalizing to 10% global firing rate, and splitting into 70/15/15 train/val/test sets."

2. **Data Statistics:**
   - "After preprocessing, the dataset contained 992 samples with 17 neurons and 100 time steps per sample (sparsity: 5.64%)."

3. **Augmentation (if used):**
   - "Training data was augmented using temporal jitter (σ=2 time steps) and spike dropout (p=0.1), increasing dataset size 3×."

4. **Quality Checks:**
   - "Data quality analysis showed mean firing rate of 0.095 ± 0.023 with no silent samples."

## 🎯 Next Steps

After preprocessing:

1. ✅ Load preprocessed data
2. ✅ Convert to PyTorch tensors
3. ✅ Create DataLoader
4. ✅ Train your SNN model
5. ✅ Evaluate on test set

```python
import torch
from torch.utils.data import TensorDataset, DataLoader

# Load
train_spikes = np.load('preprocessed_train.npy')
train_labels = np.load('preprocessed_train_labels.npy')

# Convert
dataset = TensorDataset(
    torch.FloatTensor(train_spikes),
    torch.LongTensor(train_labels)
)

loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Ready to train!
for batch_spikes, batch_labels in loader:
    # Your training code here
    pass
```

---

**You now have production-ready, preprocessed data for SNN training! 🚀**
