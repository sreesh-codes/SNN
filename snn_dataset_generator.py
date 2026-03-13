import numpy as np
from datetime import datetime
from sklearn.datasets import make_classification, make_blobs
from sklearn.preprocessing import StandardScaler

class SNNDatasetGenerator:
    """
    Generate datasets for Spiking Neural Network research on vector embeddings.
    Supports both classification and clustering tasks with spike encoding.
    
    Args:
        random_state: int or None
            - If int: Seeds the random number generator for reproducible results.
                      Use any integer for reproducibility (e.g., 42, 123, 0).
            - If None: Uses system entropy for truly random generation each run.
            
    Note:
        The choice of random_state does NOT affect dataset fairness or validity.
        It only controls reproducibility. Any seed produces an equally valid dataset.
    """
    
    def __init__(self, random_state=None):
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)
    
    def _make_rng(self):
        """
        Create a local random number generator so that spike encoding and
        temporal pattern generation are fully reproducible for a given
        random_state, independent of global NumPy RNG state.
        """
        return np.random.RandomState(self.random_state)
        
    def rate_encoding(self, values, max_time_steps=100, max_rate=100):
        """
        Convert continuous values to spike trains using rate encoding.
        Higher values = higher firing rates
        
        Args:
            values: numpy array of shape (n_samples, n_features)
            max_time_steps: number of time steps for spike train
            max_rate: maximum firing rate (Hz)
        
        Returns:
            spike_trains: binary array of shape (n_samples, n_features, max_time_steps)
        """
        n_samples, n_features = values.shape
        spike_trains = np.zeros((n_samples, n_features, max_time_steps), dtype=np.uint8)
        
        # Normalize values to [0, 1]
        scaler = StandardScaler()
        normalized = scaler.fit_transform(values)
        # Handle edge case where all values are the same (avoid division by zero)
        value_range = normalized.max() - normalized.min()
        if value_range > 1e-10:  # Check if range is significant
            normalized = (normalized - normalized.min()) / value_range
        else:
            # If all values are the same, set to 0.5 (middle value)
            normalized = np.ones_like(normalized) * 0.5
        
        # Generate spikes based on firing rate
        # Use Bernoulli process for binary spikes: P(spike) = rate * dt
        # We assume max_time_steps corresponds to 1 second (1000ms) simulation time
        dt = 1.0 / max_time_steps
        rng = self._make_rng()
        
        for i in range(n_samples):
            for j in range(n_features):
                # Probability of spike at each step
                prob = normalized[i, j] * max_rate * dt
                # Clamp probability to [0, 1]
                prob = np.clip(prob, 0, 1)
                
                # Generate spikes
                spike_trains[i, j, :] = rng.binomial(1, prob, max_time_steps)
        
        return spike_trains, normalized
    
    def temporal_encoding(self, values, max_time_steps=100):
        """
        Convert continuous values to spike trains using temporal encoding.
        Higher values = earlier spike times
        
        Args:
            values: numpy array of shape (n_samples, n_features)
            max_time_steps: number of time steps for spike train
        
        Returns:
            spike_trains: binary array of shape (n_samples, n_features, max_time_steps)
        """
        n_samples, n_features = values.shape
        spike_trains = np.zeros((n_samples, n_features, max_time_steps), dtype=np.uint8)
        
        # Normalize values to [0, 1]
        scaler = StandardScaler()
        normalized = scaler.fit_transform(values)
        # Handle edge case where all values are the same (avoid division by zero)
        value_range = normalized.max() - normalized.min()
        if value_range > 1e-10:  # Check if range is significant
            normalized = (normalized - normalized.min()) / value_range
        else:
            # If all values are the same, set to 0.5 (middle value)
            normalized = np.ones_like(normalized) * 0.5
        
        # Generate single spike at time proportional to value
        for i in range(n_samples):
            for j in range(n_features):
                spike_time = int((1 - normalized[i, j]) * (max_time_steps - 1))
                spike_trains[i, j, spike_time] = 1
        
        return spike_trains, normalized
    
    def generate_classification_dataset(self, n_samples=1000, n_features=20, 
                                       n_classes=5, n_informative=15,
                                       n_redundant=2, n_clusters_per_class=2,
                                       encoding='rate', max_time_steps=100):
        """
        Generate a classification dataset suitable for SNN embedding research.
        
        Args:
            n_samples: number of samples
            n_features: number of features
            n_classes: number of classes
            n_informative: desired number of informative features
            n_redundant: desired number of redundant features
            n_clusters_per_class: number of clusters per class
            encoding: 'rate' or 'temporal'
            max_time_steps: time steps for spike encoding
        
        Returns:
            dataset: dictionary with features, labels, spike_trains, metadata
        """
        # Ensure sklearn's constraints:
        # (1) n_informative + n_redundant + n_repeated <= n_features
        # (2) n_classes * n_clusters_per_class <= 2 ** n_informative
        # We don't expose n_repeated, so treat it as 0 here and adjust
        # n_informative / n_redundant conservatively when needed.
        # Constraint (2): how many informative bits are required to represent
        # all (class, cluster) combinations?
        if n_classes * n_clusters_per_class <= 1:
            min_inf_for_clusters = 1
        else:
            min_inf_for_clusters = int(np.ceil(np.log2(n_classes * n_clusters_per_class)))
        
        # Maximum informative features allowed by total feature budget
        max_informative = max(1, n_features - 1)  # leave at least 1 feature for redundant/repeated
        
        if max_informative < min_inf_for_clusters:
            raise ValueError(
                f"Incompatible configuration: n_features={n_features} is too small for "
                f"n_classes={n_classes} and n_clusters_per_class={n_clusters_per_class}. "
                f"Need at least n_informative={min_inf_for_clusters}, but can only allocate "
                f"{max_informative}. Increase n_features or reduce n_classes / n_clusters_per_class."
            )
        
        # Choose effective informative features within valid range
        n_informative_eff = min(
            max(n_informative, min_inf_for_clusters, 2),
            max_informative,
        )
        
        # Allocate redundant features within the remaining budget
        remaining_for_redundant = max(0, n_features - n_informative_eff)
        n_redundant_eff = min(max(0, n_redundant), max(0, remaining_for_redundant))
        
        # Generate base classification data
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative_eff,
            n_redundant=n_redundant_eff,
            n_classes=n_classes,
            n_clusters_per_class=n_clusters_per_class,
            random_state=self.random_state,
            shuffle=True
        )
        
        # Encode as spike trains
        if encoding == 'rate':
            spike_trains, normalized = self.rate_encoding(X, max_time_steps)
        elif encoding == 'temporal':
            spike_trains, normalized = self.temporal_encoding(X, max_time_steps)
        else:
            raise ValueError("Encoding must be 'rate' or 'temporal'")
        
        dataset = {
            'features': X,
            'normalized_features': normalized,
            'labels': y,
            'spike_trains': spike_trains,
            'metadata': {
                'n_samples': n_samples,
                'n_features': n_features,
                'n_classes': n_classes,
                'n_informative': n_informative,
                'encoding': encoding,
                'max_time_steps': max_time_steps,
                'task': 'classification',
                'class_distribution': {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
                'random_state': self.random_state,
                'generated_at': datetime.now().isoformat()
            }
        }
        
        return dataset
    
    def generate_clustering_dataset(self, n_samples=1000, n_features=20,
                                   n_clusters=5, cluster_std=1.0,
                                   encoding='rate', max_time_steps=100):
        """
        Generate a clustering dataset suitable for SNN embedding research.
        
        Args:
            n_samples: number of samples
            n_features: number of features
            n_clusters: number of clusters
            cluster_std: standard deviation of clusters
            encoding: 'rate' or 'temporal'
            max_time_steps: time steps for spike encoding
        
        Returns:
            dataset: dictionary with features, labels, spike_trains, metadata
        """
        # Generate base clustering data
        X, y = make_blobs(
            n_samples=n_samples,
            n_features=n_features,
            centers=n_clusters,
            cluster_std=cluster_std,
            random_state=self.random_state,
            shuffle=True
        )
        
        # Encode as spike trains
        if encoding == 'rate':
            spike_trains, normalized = self.rate_encoding(X, max_time_steps)
        elif encoding == 'temporal':
            spike_trains, normalized = self.temporal_encoding(X, max_time_steps)
        else:
            raise ValueError("Encoding must be 'rate' or 'temporal'")
        
        dataset = {
            'features': X,
            'normalized_features': normalized,
            'labels': y,  # Ground truth clusters
            'spike_trains': spike_trains,
            'metadata': {
                'n_samples': n_samples,
                'n_features': n_features,
                'n_clusters': n_clusters,
                'cluster_std': cluster_std,
                'encoding': encoding,
                'max_time_steps': max_time_steps,
                'task': 'clustering',
                'cluster_distribution': {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
                'random_state': self.random_state,
                'generated_at': datetime.now().isoformat()
            }
        }
        
        return dataset
    
    def generate_temporal_pattern_dataset(self, n_samples=1000, n_features=10,
                                         n_classes=5, pattern_length=50):
        """
        Generate temporal patterns that naturally suit SNNs.
        Each class has characteristic temporal dynamics.
        
        Args:
            n_samples: number of samples
            n_features: number of features (neurons)
            n_classes: number of pattern classes
            pattern_length: length of temporal pattern
        
        Returns:
            dataset: dictionary with temporal patterns and labels
        """
        spike_trains = np.zeros((n_samples, n_features, pattern_length), dtype=np.uint8)
        rng = self._make_rng()
        
        # Ensure balanced class distribution for fair dataset
        samples_per_class = n_samples // n_classes
        labels = np.repeat(np.arange(n_classes), samples_per_class)
        # Add any remaining samples to first classes to maintain near-perfect balance
        if len(labels) < n_samples:
            remaining = n_samples - len(labels)
            labels = np.concatenate([labels, np.arange(remaining)])
        # Shuffle to randomize order using local RNG for reproducibility
        rng.shuffle(labels)
        
        for i in range(n_samples):
            class_id = labels[i]
            
            # Create class-specific temporal patterns
            if class_id == 0:
                # Early burst pattern
                spike_trains[i, :, :10] = rng.binomial(1, 0.8, (n_features, 10))
            elif class_id == 1:
                # Late burst pattern
                spike_trains[i, :, -10:] = rng.binomial(1, 0.8, (n_features, 10))
            elif class_id == 2:
                # Oscillatory pattern
                for t in range(0, pattern_length, 10):
                    spike_trains[i, :, t:t+3] = rng.binomial(1, 0.7, (n_features, 3))
            elif class_id == 3:
                # Random sparse pattern
                spike_trains[i, :, :] = rng.binomial(1, 0.1, (n_features, pattern_length))
            else:
                # Dense uniform pattern
                spike_trains[i, :, :] = rng.binomial(1, 0.4, (n_features, pattern_length))
        
        dataset = {
            'spike_trains': spike_trains,
            'labels': labels,
            'metadata': {
                'n_samples': n_samples,
                'n_features': n_features,
                'n_classes': n_classes,
                'pattern_length': pattern_length,
                'task': 'temporal_classification',
                'pattern_types': ['early_burst', 'late_burst', 'oscillatory', 'sparse', 'dense'],
                'class_distribution': {int(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))},
                'random_state': self.random_state,
                'generated_at': datetime.now().isoformat()
            }
        }
        
        return dataset
