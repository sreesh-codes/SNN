# SNN Dataset Generator for Vector Embeddings
## Custom Datasets for Spiking Neural Network Research

This repository contains tools for generating datasets specifically designed for researching vector embeddings with Spiking Neural Networks (SNNs).

## 🚀 Quick Start

### Run the Dashboard

The easiest way to generate and visualize datasets is using the interactive dashboard:

```bash
streamlit run dashboard.py
```

This will open a web interface where you can:
1. Configure dataset parameters (samples, features, classes, etc.)
2. Visualize spike trains
3. Download generated datasets as ZIP files

### Programmatic Usage

You can also use the generator directly in your Python scripts:

```python
from snn_dataset_generator import SNNDatasetGenerator

# Initialize generator
generator = SNNDatasetGenerator(random_state=42)

# Create classification dataset
dataset = generator.generate_classification_dataset(
    n_samples=500,
    n_features=15,
    n_classes=3,
    encoding='rate',
    max_time_steps=80
)

# Access data
spike_trains = dataset['spike_trains']
labels = dataset['labels']

print(f"Shape: {spike_trains.shape}")  # (n_samples, n_features, time_steps)
```

## 📊 Supported Dataset Types

1. **Classification Dataset (Rate Encoding)**
   - Features encoded as spike rates (Poisson process)
   - Higher values = higher firing rates

2. **Classification Dataset (Temporal Encoding)**
   - Features encoded by spike timing
   - Higher values = earlier spikes (latency coding)

3. **Clustering Dataset**
   - Generated using Gaussian blobs
   - Ideal for unsupervised learning tasks

4. **Temporal Pattern Dataset**
   - Distinct temporal patterns for each class
   - Patterns: early burst, late burst, oscillatory, sparse, dense

## 🧠 Encoding Schemes

### 1. Rate Encoding
- Higher feature values → higher firing rates
- Spikes generated via Poisson process
- More biologically plausible
- Better for continuous data

### 2. Temporal Encoding
- Higher feature values → earlier spike times
- Single spike per neuron per sample
- More efficient (fewer spikes)
- Better for rapid classification

## 🔬 Research Applications

### 1. Vector Embedding Learning
Use these datasets to train SNNs that learn embeddings for:
- Classification tasks
- Clustering/unsupervised learning
- Dimensionality reduction
- Feature extraction

### 2. Temporal Pattern Recognition
The temporal pattern dataset is ideal for:
- Studying STDP (Spike-Timing-Dependent Plasticity)
- Reservoir computing
- Liquid state machines
- Event-based processing

### 3. Energy Efficiency Research
Compare embeddings learned by:
- Traditional ANNs (backprop)
- SNNs with various learning rules
- Hybrid approaches

## 🛠️ SNN Framework Integration

### snnTorch Example
```python
import snntorch as snn
import torch.nn as nn
import torch

class SNNEmbedding(nn.Module):
    def __init__(self, n_features=20, n_hidden=100, n_classes=5):
        super().__init__()
        self.fc1 = nn.Linear(n_features, n_hidden)
        self.lif1 = snn.Leaky(beta=0.9)
        self.fc2 = nn.Linear(n_hidden, n_classes)
        self.lif2 = snn.Leaky(beta=0.9)
    
    def forward(self, x):
        # x shape: (batch, n_features, time_steps)
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        
        spk_rec = []
        for step in range(x.shape[2]):
            cur1 = self.fc1(x[:, :, step])
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk_rec.append(spk2)
        
        return torch.stack(spk_rec, dim=2)
```

## 🤝 Contributing

Want to add new encoding schemes or dataset types? The generator is modular:

```python
class SNNDatasetGenerator:
    def my_custom_encoding(self, values):
        # Your encoding logic
        return spike_trains
    
    def generate_my_custom_dataset(self):
        # Your dataset generation
        return dataset
```

## 📧 Contact

For questions or collaboration on SNN embedding research, feel free to reach out!

## 📄 License

MIT License - Use freely for research and commercial applications.

---

**Happy Spiking! 🧠⚡**
