"""
SNN Dataset Preprocessing Module
==================================

Comprehensive preprocessing pipeline for spike train datasets:
- Noise filtering
- Spike rate normalization
- Temporal augmentation
- Train/validation/test splitting
- Batch preparation
- Data quality analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import json
import os
from typing import Tuple, Dict, List, Optional
import warnings
from snn_dataset_generator import SNNDatasetGenerator
warnings.filterwarnings('ignore')


class SNNDataPreprocessor:
    """
    Preprocessing pipeline for SNN spike train datasets.
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(random_state)
        self.scaler = None
        self.label_encoder = None
        self.preprocessing_stats = {}
    
    # ========================================================================
    # 1. DATA LOADING AND VALIDATION
    # ========================================================================
    
    def load_dataset(self, dataset_name: str, base_path: str = 'snn_datasets') -> Dict:
        """
        Load dataset with validation.
        
        Args:
            dataset_name: Name of the dataset (e.g., 'classification_rate')
            base_path: Directory containing datasets
        
        Returns:
            Dictionary with spike_trains, labels, features, metadata
        """
        print(f"Loading dataset: {dataset_name}")
        
        try:
            # Load spike trains
            spike_trains = np.load(f'{base_path}/{dataset_name}_spike_trains.npy')
            labels = np.load(f'{base_path}/{dataset_name}_labels.npy')
            
            # Try to load features if available
            try:
                features = np.load(f'{base_path}/{dataset_name}_features.npy')
                normalized_features = np.load(f'{base_path}/{dataset_name}_normalized_features.npy')
            except FileNotFoundError:
                features = None
                normalized_features = None
            
            # Load metadata
            with open(f'{base_path}/{dataset_name}_metadata.json', 'r') as f:
                metadata = json.load(f)
        
        except FileNotFoundError:
            # If raw files do not exist, generate a fresh dataset using the generator
            print("✗ Raw dataset files not found on disk.")
            print("  → Generating dataset using SNNDatasetGenerator with default parameters.")
            os.makedirs(base_path, exist_ok=True)
            
            gen = SNNDatasetGenerator(random_state=self.random_state)
            
            if dataset_name.startswith('classification'):
                generated = gen.generate_classification_dataset()
            elif dataset_name.startswith('clustering'):
                # Default clustering configuration
                generated = gen.generate_clustering_dataset()
            elif dataset_name.startswith('temporal'):
                generated = gen.generate_temporal_pattern_dataset()
            else:
                raise ValueError(
                    f"Unknown dataset_name '{dataset_name}'. "
                    "Expected to start with 'classification', 'clustering', or 'temporal'."
                )
            
            spike_trains = generated['spike_trains']
            labels = generated['labels']
            features = generated.get('features', None)
            normalized_features = generated.get('normalized_features', None)
            metadata = generated['metadata']
            
            # Persist generated dataset for future runs
            np.save(f'{base_path}/{dataset_name}_spike_trains.npy', spike_trains)
            np.save(f'{base_path}/{dataset_name}_labels.npy', labels)
            if features is not None:
                np.save(f'{base_path}/{dataset_name}_features.npy', features)
            if normalized_features is not None:
                np.save(f'{base_path}/{dataset_name}_normalized_features.npy', normalized_features)
            with open(f'{base_path}/{dataset_name}_metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print("  ✓ Generated and saved new raw dataset files.")
        
        except Exception as e:
            print(f"✗ Error loading or generating dataset: {e}")
            raise
        
        dataset = {
            'spike_trains': spike_trains,
            'labels': labels,
            'features': features,
            'normalized_features': normalized_features,
            'metadata': metadata,
            'name': dataset_name
        }
        
        # Validate
        self._validate_dataset(dataset)
        
        print(f"✓ Dataset ready")
        print(f"  Shape: {spike_trains.shape}")
        print(f"  Classes: {len(np.unique(labels))}")
        print(f"  Total spikes: {spike_trains.sum():,.0f}")
        
        return dataset
    
    def _validate_dataset(self, dataset: Dict):
        """Validate dataset integrity."""
        spike_trains = dataset['spike_trains']
        labels = dataset['labels']
        
        # Check shapes
        assert len(spike_trains) == len(labels), "Mismatch between samples and labels"
        
        # Check spike values (should be 0 or 1)
        unique_vals = np.unique(spike_trains)
        assert set(unique_vals).issubset({0, 1}), f"Invalid spike values: {unique_vals}"
        
        # Check for NaN or Inf
        assert not np.isnan(spike_trains).any(), "Dataset contains NaN values"
        assert not np.isinf(spike_trains).any(), "Dataset contains Inf values"
        
        print("✓ Dataset validation passed")
    
    # ========================================================================
    # 2. NOISE FILTERING
    # ========================================================================
    
    def remove_noisy_neurons(self, spike_trains: np.ndarray, 
                            threshold: float = 0.01) -> Tuple[np.ndarray, List[int]]:
        """
        Remove neurons with very low or very high firing rates (likely noise).
        
        Args:
            spike_trains: Shape (n_samples, n_neurons, time_steps)
            threshold: Remove neurons with firing rate < threshold or > (1-threshold)
        
        Returns:
            Filtered spike trains, indices of kept neurons
        """
        n_samples, n_neurons, time_steps = spike_trains.shape
        
        # Calculate firing rate per neuron (across all samples)
        firing_rates = spike_trains.sum(axis=(0, 2)) / (n_samples * time_steps)
        
        # Identify good neurons
        good_neurons = np.where(
            (firing_rates >= threshold) & 
            (firing_rates <= (1 - threshold))
        )[0]
        
        # Filter
        filtered_trains = spike_trains[:, good_neurons, :]
        
        removed = n_neurons - len(good_neurons)
        print(f"Noise filtering: Removed {removed} neurons ({removed/n_neurons*100:.1f}%)")
        print(f"  Kept {len(good_neurons)} neurons")
        
        self.preprocessing_stats['noisy_neurons_removed'] = removed
        self.preprocessing_stats['kept_neurons'] = good_neurons.tolist()
        
        return filtered_trains, good_neurons
    
    def remove_silent_samples(self, spike_trains: np.ndarray, labels: np.ndarray,
                             min_spikes: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove samples with too few spikes (likely corrupted).
        
        Args:
            spike_trains: Shape (n_samples, n_neurons, time_steps)
            labels: Shape (n_samples,)
            min_spikes: Minimum total spikes per sample
        
        Returns:
            Filtered spike trains and labels
        """
        # Calculate spikes per sample
        spikes_per_sample = spike_trains.sum(axis=(1, 2))
        
        # Identify good samples
        good_samples = np.where(spikes_per_sample >= min_spikes)[0]
        
        # Filter
        filtered_trains = spike_trains[good_samples]
        filtered_labels = labels[good_samples]
        
        removed = len(spike_trains) - len(good_samples)
        print(f"Silent sample removal: Removed {removed} samples ({removed/len(spike_trains)*100:.1f}%)")
        print(f"  Kept {len(good_samples)} samples")
        
        self.preprocessing_stats['silent_samples_removed'] = removed
        
        return filtered_trains, filtered_labels
    
    # ========================================================================
    # 3. NORMALIZATION
    # ========================================================================
    
    def normalize_spike_rates(self, spike_trains: np.ndarray,
                             method: str = 'global') -> np.ndarray:
        """
        Normalize firing rates across the dataset.
        
        Args:
            spike_trains: Shape (n_samples, n_neurons, time_steps)
            method: 'global', 'per_neuron', or 'per_sample'
        
        Returns:
            Normalized spike trains (still binary)
        """
        if method == 'global':
            # Scale based on global mean firing rate
            target_rate = 0.1  # Target 10% firing rate
            current_rate = spike_trains.mean()
            
            if current_rate > 0:
                scale_factor = target_rate / current_rate
                # Probabilistic scaling
                normalized = np.random.binomial(1, 
                    np.clip(spike_trains * scale_factor, 0, 1))
            else:
                normalized = spike_trains
            
            print(f"Global normalization: {current_rate:.3f} → {normalized.mean():.3f}")
        
        elif method == 'per_neuron':
            # Normalize each neuron independently
            normalized = spike_trains.copy()
            n_samples, n_neurons, time_steps = spike_trains.shape
            
            for neuron_idx in range(n_neurons):
                neuron_spikes = spike_trains[:, neuron_idx, :]
                current_rate = neuron_spikes.mean()
                
                if current_rate > 0:
                    target_rate = 0.1
                    scale_factor = target_rate / current_rate
                    normalized[:, neuron_idx, :] = np.random.binomial(1,
                        np.clip(neuron_spikes * scale_factor, 0, 1))
            
            print(f"Per-neuron normalization completed")
        
        elif method == 'per_sample':
            # Normalize each sample independently
            normalized = spike_trains.copy()
            
            for sample_idx in range(len(spike_trains)):
                sample_spikes = spike_trains[sample_idx]
                current_rate = sample_spikes.mean()
                
                if current_rate > 0:
                    target_rate = 0.1
                    scale_factor = target_rate / current_rate
                    normalized[sample_idx] = np.random.binomial(1,
                        np.clip(sample_spikes * scale_factor, 0, 1))
            
            print(f"Per-sample normalization completed")
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        self.preprocessing_stats['normalization_method'] = method
        
        return normalized
    
    def balance_classes(self, spike_trains: np.ndarray, labels: np.ndarray,
                       method: str = 'undersample') -> Tuple[np.ndarray, np.ndarray]:
        """
        Balance class distribution.
        
        Args:
            spike_trains: Shape (n_samples, n_neurons, time_steps)
            labels: Shape (n_samples,)
            method: 'undersample' or 'oversample'
        
        Returns:
            Balanced spike trains and labels
        """
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        if method == 'undersample':
            # Undersample to smallest class
            min_count = counts.min()
            
            balanced_indices = []
            for label in unique_labels:
                label_indices = np.where(labels == label)[0]
                selected = np.random.choice(label_indices, size=min_count, replace=False)
                balanced_indices.extend(selected)
            
            balanced_indices = np.array(balanced_indices)
            np.random.shuffle(balanced_indices)
            
            balanced_trains = spike_trains[balanced_indices]
            balanced_labels = labels[balanced_indices]
            
            print(f"Undersampling: {len(spike_trains)} → {len(balanced_trains)} samples")
        
        elif method == 'oversample':
            # Oversample to largest class
            max_count = counts.max()
            
            balanced_trains = []
            balanced_labels = []
            
            for label in unique_labels:
                label_indices = np.where(labels == label)[0]
                label_data = spike_trains[label_indices]
                
                # Oversample with replacement
                oversampled_indices = np.random.choice(
                    len(label_data), size=max_count, replace=True
                )
                balanced_trains.append(label_data[oversampled_indices])
                balanced_labels.extend([label] * max_count)
            
            balanced_trains = np.vstack(balanced_trains)
            balanced_labels = np.array(balanced_labels)
            
            # Shuffle
            shuffle_idx = np.random.permutation(len(balanced_labels))
            balanced_trains = balanced_trains[shuffle_idx]
            balanced_labels = balanced_labels[shuffle_idx]
            
            print(f"Oversampling: {len(spike_trains)} → {len(balanced_trains)} samples")
        
        else:
            raise ValueError(f"Unknown balancing method: {method}")
        
        # Show new distribution
        unique_new, counts_new = np.unique(balanced_labels, return_counts=True)
        print(f"  New distribution: {dict(zip(unique_new, counts_new))}")
        
        self.preprocessing_stats['balancing_method'] = method
        
        return balanced_trains, balanced_labels
    
    # ========================================================================
    # 4. DATA AUGMENTATION
    # ========================================================================
    
    def augment_jitter(self, spike_trains: np.ndarray, 
                      jitter_std: float = 2.0) -> np.ndarray:
        """
        Add temporal jitter to spikes (shift spike times slightly).
        
        Args:
            spike_trains: Shape (n_samples, n_neurons, time_steps)
            jitter_std: Standard deviation of jitter in time steps
        
        Returns:
            Augmented spike trains
        """
        n_samples, n_neurons, time_steps = spike_trains.shape
        augmented = np.zeros_like(spike_trains)
        
        for i in range(n_samples):
            for j in range(n_neurons):
                # Find spike times
                spike_times = np.where(spike_trains[i, j] == 1)[0]
                
                # Add jitter
                jittered_times = spike_times + np.random.normal(0, jitter_std, len(spike_times))
                jittered_times = np.clip(jittered_times, 0, time_steps - 1).astype(int)
                
                # Set spikes at jittered times
                augmented[i, j, jittered_times] = 1
        
        print(f"Temporal jitter augmentation: std={jitter_std}")
        
        return augmented
    
    def augment_dropout(self, spike_trains: np.ndarray, 
                       dropout_rate: float = 0.1) -> np.ndarray:
        """
        Randomly drop spikes (spike dropout).
        
        Args:
            spike_trains: Shape (n_samples, n_neurons, time_steps)
            dropout_rate: Probability of dropping each spike
        
        Returns:
            Augmented spike trains
        """
        # Create dropout mask
        mask = np.random.binomial(1, 1 - dropout_rate, spike_trains.shape)
        augmented = spike_trains * mask
        
        original_spikes = spike_trains.sum()
        augmented_spikes = augmented.sum()
        dropped = original_spikes - augmented_spikes
        
        print(f"Spike dropout: Dropped {dropped:,.0f} spikes ({dropped/original_spikes*100:.1f}%)")
        
        return augmented
    
    def augment_noise(self, spike_trains: np.ndarray,
                     noise_rate: float = 0.01) -> np.ndarray:
        """
        Add random noise spikes.
        
        Args:
            spike_trains: Shape (n_samples, n_neurons, time_steps)
            noise_rate: Probability of adding noise spike at each position
        
        Returns:
            Augmented spike trains
        """
        # Generate noise
        noise = np.random.binomial(1, noise_rate, spike_trains.shape)
        
        # Add noise (logical OR)
        augmented = np.clip(spike_trains + noise, 0, 1)
        
        added_spikes = augmented.sum() - spike_trains.sum()
        
        print(f"Noise augmentation: Added {added_spikes:,.0f} noise spikes")
        
        return augmented
    
    def augment_time_warp(self, spike_trains: np.ndarray,
                         warp_factor: float = 0.1) -> np.ndarray:
        """
        Apply time warping (stretch/compress temporal dimension).
        
        Args:
            spike_trains: Shape (n_samples, n_neurons, time_steps)
            warp_factor: Maximum warping factor (0.1 = ±10%)
        
        Returns:
            Augmented spike trains
        """
        n_samples, n_neurons, time_steps = spike_trains.shape
        augmented = np.zeros_like(spike_trains)
        
        for i in range(n_samples):
            # Random warp factor for this sample
            warp = 1.0 + np.random.uniform(-warp_factor, warp_factor)
            
            for j in range(n_neurons):
                spike_times = np.where(spike_trains[i, j] == 1)[0]
                
                # Warp time
                warped_times = (spike_times * warp).astype(int)
                warped_times = np.clip(warped_times, 0, time_steps - 1)
                
                augmented[i, j, warped_times] = 1
        
        print(f"Time warp augmentation: factor=±{warp_factor*100:.0f}%")
        
        return augmented
    
    # ========================================================================
    # 5. TRAIN/VAL/TEST SPLITTING
    # ========================================================================
    
    def split_dataset(self, spike_trains: np.ndarray, labels: np.ndarray,
                     train_ratio: float = 0.7,
                     val_ratio: float = 0.15,
                     test_ratio: float = 0.15,
                     stratify: bool = True) -> Dict:
        """
        Split dataset into train/validation/test sets.
        
        Args:
            spike_trains: Shape (n_samples, n_neurons, time_steps)
            labels: Shape (n_samples,)
            train_ratio: Fraction for training
            val_ratio: Fraction for validation
            test_ratio: Fraction for testing
            stratify: Maintain class distribution in splits
        
        Returns:
            Dictionary with train/val/test splits
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"
        
        # First split: train vs (val + test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            spike_trains, labels,
            test_size=(val_ratio + test_ratio),
            random_state=self.random_state,
            stratify=labels if stratify else None
        )
        
        # Second split: val vs test
        val_size = val_ratio / (val_ratio + test_ratio)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=(1 - val_size),
            random_state=self.random_state,
            stratify=y_temp if stratify else None
        )
        
        splits = {
            'train': {'spike_trains': X_train, 'labels': y_train},
            'val': {'spike_trains': X_val, 'labels': y_val},
            'test': {'spike_trains': X_test, 'labels': y_test}
        }
        
        print(f"\nDataset split:")
        print(f"  Train: {len(y_train)} samples ({len(y_train)/len(labels)*100:.1f}%)")
        print(f"  Val:   {len(y_val)} samples ({len(y_val)/len(labels)*100:.1f}%)")
        print(f"  Test:  {len(y_test)} samples ({len(y_test)/len(labels)*100:.1f}%)")
        
        if stratify:
            print("\n  Class distribution (train/val/test):")
            for cls in np.unique(labels):
                train_count = (y_train == cls).sum()
                val_count = (y_val == cls).sum()
                test_count = (y_test == cls).sum()
                print(f"    Class {cls}: {train_count}/{val_count}/{test_count}")
        
        self.preprocessing_stats['split_ratios'] = {
            'train': train_ratio,
            'val': val_ratio,
            'test': test_ratio
        }
        
        return splits
    
    # ========================================================================
    # 6. DATA QUALITY ANALYSIS
    # ========================================================================
    
    def analyze_data_quality(self, spike_trains: np.ndarray, labels: np.ndarray,
                            save_path: Optional[str] = None):
        """
        Comprehensive data quality analysis with visualizations.
        
        Args:
            spike_trains: Shape (n_samples, n_neurons, time_steps)
            labels: Shape (n_samples,)
            save_path: Optional path to save visualizations
        """
        n_samples, n_neurons, time_steps = spike_trains.shape
        
        print("\n" + "="*80)
        print("DATA QUALITY ANALYSIS")
        print("="*80)
        
        # 1. Basic statistics
        total_spikes = spike_trains.sum()
        spikes_per_sample = spike_trains.sum(axis=(1, 2))
        spikes_per_neuron = spike_trains.sum(axis=(0, 2))
        spikes_per_timestep = spike_trains.sum(axis=(0, 1))
        
        print(f"\nBasic Statistics:")
        print(f"  Total spikes: {total_spikes:,.0f}")
        print(f"  Sparsity: {(total_spikes / spike_trains.size) * 100:.2f}%")
        print(f"  Spikes per sample: {spikes_per_sample.mean():.2f} ± {spikes_per_sample.std():.2f}")
        print(f"  Min/Max spikes per sample: {spikes_per_sample.min():.0f} / {spikes_per_sample.max():.0f}")
        
        # 2. Firing rate analysis
        firing_rate_per_neuron = spikes_per_neuron / (n_samples * time_steps)
        
        print(f"\nFiring Rate Analysis:")
        print(f"  Mean firing rate: {firing_rate_per_neuron.mean():.4f}")
        print(f"  Std firing rate: {firing_rate_per_neuron.std():.4f}")
        print(f"  Silent neurons: {(firing_rate_per_neuron == 0).sum()}")
        print(f"  Highly active neurons (>50%): {(firing_rate_per_neuron > 0.5).sum()}")
        
        # 3. Temporal analysis
        print(f"\nTemporal Analysis:")
        print(f"  Active timesteps: {(spikes_per_timestep > 0).sum()} / {time_steps}")
        temporal_sparsity = (spikes_per_timestep == 0).sum() / time_steps * 100
        print(f"  Temporal sparsity: {temporal_sparsity:.2f}%")
        
        # 4. Class-wise analysis
        print(f"\nClass-wise Analysis:")
        for cls in np.unique(labels):
            cls_samples = spike_trains[labels == cls]
            cls_spikes = cls_samples.sum(axis=(1, 2))
            print(f"  Class {cls}: {len(cls_samples)} samples, "
                  f"avg spikes = {cls_spikes.mean():.2f} ± {cls_spikes.std():.2f}")
        
        # 5. Data quality issues
        print(f"\nPotential Issues:")
        silent_samples = (spikes_per_sample == 0).sum()
        low_spike_samples = (spikes_per_sample < 10).sum()
        high_spike_samples = (spikes_per_sample > spikes_per_sample.mean() + 3*spikes_per_sample.std()).sum()
        
        if silent_samples > 0:
            print(f"  ⚠ {silent_samples} completely silent samples")
        if low_spike_samples > 0:
            print(f"  ⚠ {low_spike_samples} samples with <10 spikes")
        if high_spike_samples > 0:
            print(f"  ⚠ {high_spike_samples} outlier samples (>3σ)")
        
        if silent_samples == 0 and low_spike_samples == 0:
            print(f"  ✓ No major quality issues detected")
        
        # 6. Visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Spikes per sample histogram
        ax = axes[0, 0]
        ax.hist(spikes_per_sample, bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(spikes_per_sample.mean(), color='red', linestyle='--', 
                   label=f'Mean: {spikes_per_sample.mean():.0f}')
        ax.set_xlabel('Total Spikes per Sample')
        ax.set_ylabel('Frequency')
        ax.set_title('Spike Distribution per Sample')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Firing rate per neuron
        ax = axes[0, 1]
        ax.bar(range(n_neurons), firing_rate_per_neuron)
        ax.axhline(firing_rate_per_neuron.mean(), color='red', linestyle='--',
                   label=f'Mean: {firing_rate_per_neuron.mean():.3f}')
        ax.set_xlabel('Neuron Index')
        ax.set_ylabel('Firing Rate')
        ax.set_title('Firing Rate per Neuron')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Temporal dynamics
        ax = axes[0, 2]
        ax.plot(spikes_per_timestep)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Total Spikes')
        ax.set_title('Temporal Spike Distribution')
        ax.grid(True, alpha=0.3)
        
        # Class distribution
        ax = axes[1, 0]
        unique, counts = np.unique(labels, return_counts=True)
        ax.bar(unique, counts, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Class')
        ax.set_ylabel('Sample Count')
        ax.set_title('Class Distribution')
        ax.set_xticks(unique)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Spikes per class (boxplot)
        ax = axes[1, 1]
        class_spike_data = [spikes_per_sample[labels == cls] for cls in unique]
        bp = ax.boxplot(class_spike_data, labels=unique, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        ax.set_xlabel('Class')
        ax.set_ylabel('Spikes per Sample')
        ax.set_title('Spike Distribution by Class')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Correlation heatmap (small sample)
        ax = axes[1, 2]
        if n_neurons <= 20:
            # Compute correlation between neurons
            neuron_data = spike_trains.reshape(n_samples, -1)[:, :n_neurons*min(10, time_steps)]
            corr = np.corrcoef(neuron_data.T)[:n_neurons, :n_neurons]
            im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
            ax.set_xlabel('Neuron Index')
            ax.set_ylabel('Neuron Index')
            ax.set_title('Neuron Correlation Matrix')
            plt.colorbar(im, ax=ax)
        else:
            ax.text(0.5, 0.5, 'Too many neurons\nfor correlation plot',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Neuron Correlation Matrix')
        
        plt.suptitle('Data Quality Analysis Report', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n✓ Visualization saved to: {save_path}")
        
        plt.show()
        
        return {
            'total_spikes': int(total_spikes),
            'sparsity': float((total_spikes / spike_trains.size) * 100),
            'mean_spikes_per_sample': float(spikes_per_sample.mean()),
            'std_spikes_per_sample': float(spikes_per_sample.std()),
            'mean_firing_rate': float(firing_rate_per_neuron.mean()),
            'silent_samples': int(silent_samples),
            'low_spike_samples': int(low_spike_samples)
        }
    
    # ========================================================================
    # 7. COMPLETE PREPROCESSING PIPELINE
    # ========================================================================
    
    def preprocess_pipeline(self, dataset_name: str,
                           remove_noise: bool = True,
                           normalize: bool = True,
                           balance: bool = False,
                           augment: bool = False,
                           split: bool = True,
                           save_processed: bool = True) -> Dict:
        """
        Complete preprocessing pipeline.
        
        Args:
            dataset_name: Name of dataset to load
            remove_noise: Remove noisy neurons and silent samples
            normalize: Normalize firing rates
            balance: Balance class distribution
            augment: Apply data augmentation
            split: Split into train/val/test
            save_processed: Save processed data
        
        Returns:
            Preprocessed dataset dictionary
        """
        print("\n" + "="*80)
        print("SNN DATA PREPROCESSING PIPELINE")
        print("="*80 + "\n")
        
        # 1. Load dataset
        dataset = self.load_dataset(dataset_name)
        spike_trains = dataset['spike_trains']
        labels = dataset['labels']
        
        original_shape = spike_trains.shape
        
        # 2. Remove noise
        if remove_noise:
            print("\n--- Step 1: Noise Removal ---")
            spike_trains, kept_neurons = self.remove_noisy_neurons(spike_trains)
            spike_trains, labels = self.remove_silent_samples(spike_trains, labels)
        
        # 3. Normalize
        if normalize:
            print("\n--- Step 2: Normalization ---")
            spike_trains = self.normalize_spike_rates(spike_trains, method='global')
        
        # 4. Balance classes
        if balance:
            print("\n--- Step 3: Class Balancing ---")
            spike_trains, labels = self.balance_classes(spike_trains, labels, 
                                                       method='undersample')
        
        # 5. Augment
        if augment:
            print("\n--- Step 4: Data Augmentation ---")
            # Create augmented copies
            augmented_data = []
            augmented_labels = []
            
            # Original data
            augmented_data.append(spike_trains)
            augmented_labels.append(labels)
            
            # Jittered version
            jittered = self.augment_jitter(spike_trains, jitter_std=2.0)
            augmented_data.append(jittered)
            augmented_labels.append(labels)
            
            # Dropout version
            dropout = self.augment_dropout(spike_trains, dropout_rate=0.1)
            augmented_data.append(dropout)
            augmented_labels.append(labels)
            
            # Combine
            spike_trains = np.vstack(augmented_data)
            labels = np.concatenate(augmented_labels)
            
            print(f"  Total after augmentation: {len(labels)} samples (3x original)")
        
        # 6. Split dataset
        if split:
            print("\n--- Step 5: Train/Val/Test Split ---")
            splits = self.split_dataset(spike_trains, labels)
        else:
            splits = {
                'all': {'spike_trains': spike_trains, 'labels': labels}
            }
        
        # 7. Quality analysis
        print("\n--- Step 6: Quality Analysis ---")
        quality_stats = self.analyze_data_quality(
            splits['train']['spike_trains'] if split else spike_trains,
            splits['train']['labels'] if split else labels,
            save_path=f'preprocessed_{dataset_name}_quality.png'
        )
        
        # 8. Save processed data
        if save_processed:
            print("\n--- Step 7: Saving Processed Data ---")
            self._save_processed_data(dataset_name, splits, quality_stats)
        
        # Summary
        print("\n" + "="*80)
        print("PREPROCESSING COMPLETE")
        print("="*80)
        print(f"\nOriginal shape: {original_shape}")
        print(f"Final shape: {spike_trains.shape}")
        print(f"\nPreprocessing statistics saved to preprocessing_stats.json")
        
        return {
            'splits': splits,
            'quality_stats': quality_stats,
            'preprocessing_stats': self.preprocessing_stats,
            'metadata': dataset['metadata']
        }
    
    def _save_processed_data(self, dataset_name: str, splits: Dict, quality_stats: Dict):
        """Save preprocessed data to disk."""
        import os
        output_dir = f'preprocessed_{dataset_name}'
        os.makedirs(output_dir, exist_ok=True)
        
        # Save splits
        for split_name, split_data in splits.items():
            np.save(f'{output_dir}/{split_name}_spike_trains.npy', 
                   split_data['spike_trains'])
            np.save(f'{output_dir}/{split_name}_labels.npy', 
                   split_data['labels'])
        
        # Save statistics
        stats = {
            'preprocessing_stats': self.preprocessing_stats,
            'quality_stats': quality_stats
        }
        
        with open(f'{output_dir}/preprocessing_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"  ✓ Processed data saved to: {output_dir}/")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def main():
    """Example preprocessing workflow."""
    
    # Initialize preprocessor
    preprocessor = SNNDataPreprocessor(random_state=42)
    
    # Run complete pipeline
    result = preprocessor.preprocess_pipeline(
        dataset_name='classification_rate',
        remove_noise=True,
        normalize=True,
        balance=False,  # Set to True if classes are imbalanced
        augment=False,  # Set to True for data augmentation
        split=True,
        save_processed=True
    )
    
    # Access processed data
    train_data = result['splits']['train']
    val_data = result['splits']['val']
    test_data = result['splits']['test']
    
    print("\n✓ Preprocessing complete! Ready for training.")
    print(f"\nTo load processed data:")
    print(f"  train_spikes = np.load('preprocessed_classification_rate/train_spike_trains.npy')")
    print(f"  train_labels = np.load('preprocessed_classification_rate/train_labels.npy')")


if __name__ == "__main__":
    main()
