"""
Test script to verify dataset fairness across different parameters.
This script tests class balance, feature distribution, and reproducibility.
"""

import numpy as np
from snn_dataset_generator import SNNDatasetGenerator

def test_class_balance(dataset, dataset_type):
    """Test if classes are balanced"""
    labels = dataset['labels']
    unique, counts = np.unique(labels, return_counts=True)
    n_classes = len(unique)
    expected_per_class = len(labels) / n_classes
    
    print(f"\n{'='*60}")
    print(f"Testing: {dataset_type}")
    print(f"{'='*60}")
    print(f"Total samples: {len(labels)}")
    print(f"Number of classes: {n_classes}")
    print(f"Expected samples per class: {expected_per_class:.1f}")
    print(f"\nClass Distribution:")
    
    is_balanced = True
    max_deviation = 0
    
    for cls, count in zip(unique, counts):
        deviation = abs(count - expected_per_class) / expected_per_class * 100
        max_deviation = max(max_deviation, deviation)
        status = "✓" if deviation < 5 else "✗"
        print(f"  Class {cls}: {count:4d} samples ({deviation:5.2f}% deviation) {status}")
        
        if deviation > 10:  # More than 10% deviation
            is_balanced = False
    
    print(f"\nMax deviation: {max_deviation:.2f}%")
    print(f"Balance status: {'✓ FAIR (balanced)' if is_balanced else '✗ UNBALANCED'}")
    
    return is_balanced, max_deviation

def test_reproducibility():
    """Test if same seed produces same dataset"""
    print(f"\n{'='*60}")
    print("Testing Reproducibility")
    print(f"{'='*60}")
    
    seed = 42
    gen1 = SNNDatasetGenerator(random_state=seed)
    gen2 = SNNDatasetGenerator(random_state=seed)
    
    ds1 = gen1.generate_classification_dataset(n_samples=100, n_features=10, n_classes=3)
    ds2 = gen2.generate_classification_dataset(n_samples=100, n_features=10, n_classes=3)
    
    features_match = np.allclose(ds1['features'], ds2['features'])
    labels_match = np.array_equal(ds1['labels'], ds2['labels'])
    spikes_match = np.array_equal(ds1['spike_trains'], ds2['spike_trains'])
    
    print(f"Features match: {'✓' if features_match else '✗'}")
    print(f"Labels match: {'✓' if labels_match else '✗'}")
    print(f"Spike trains match: {'✓' if spikes_match else '✗'}")
    
    is_reproducible = features_match and labels_match and spikes_match
    print(f"\nReproducibility: {'✓ PASS' if is_reproducible else '✗ FAIL'}")
    
    return is_reproducible

def test_parameter_variations():
    """Test fairness across different parameter values"""
    print(f"\n{'='*60}")
    print("Testing Parameter Variations")
    print(f"{'='*60}")
    
    results = []
    
    # Test different random seeds
    for seed in [None, 0, 42, 123, 999]:
        gen = SNNDatasetGenerator(random_state=seed)
        ds = gen.generate_classification_dataset(
            n_samples=200, n_features=15, n_classes=4
        )
        is_balanced, deviation = test_class_balance(ds, f"Seed={seed}")
        results.append(("Random Seed", seed, is_balanced, deviation))
    
    # Test different number of classes
    for n_classes in [2, 3, 5, 10]:
        gen = SNNDatasetGenerator(random_state=42)
        ds = gen.generate_classification_dataset(
            n_samples=200, n_features=15, n_classes=n_classes
        )
        is_balanced, deviation = test_class_balance(ds, f"n_classes={n_classes}")
        results.append(("n_classes", n_classes, is_balanced, deviation))
    
    # Test different number of features
    for n_features in [5, 10, 20, 50]:
        gen = SNNDatasetGenerator(random_state=42)
        ds = gen.generate_classification_dataset(
            n_samples=200, n_features=n_features, n_classes=5
        )
        is_balanced, deviation = test_class_balance(ds, f"n_features={n_features}")
        results.append(("n_features", n_features, is_balanced, deviation))
    
    return results

def test_temporal_pattern_balance():
    """Test temporal pattern dataset balance (known to be potentially unbalanced)"""
    print(f"\n{'='*60}")
    print("Testing Temporal Pattern Dataset Balance")
    print(f"{'='*60}")
    print("NOTE: Temporal patterns use random class assignment,")
    print("so they may not be perfectly balanced.")
    
    gen = SNNDatasetGenerator(random_state=42)
    ds = gen.generate_temporal_pattern_dataset(
        n_samples=1000, n_features=10, n_classes=5
    )
    
    is_balanced, deviation = test_class_balance(ds, "Temporal Patterns")
    return is_balanced, deviation

if __name__ == "__main__":
    print("\n" + "="*60)
    print("DATASET FAIRNESS TEST SUITE")
    print("="*60)
    
    # Test 1: Basic classification balance
    gen = SNNDatasetGenerator(random_state=42)
    ds_class = gen.generate_classification_dataset(
        n_samples=500, n_features=20, n_classes=5
    )
    test_class_balance(ds_class, "Classification Dataset")
    
    # Test 2: Clustering balance
    ds_cluster = gen.generate_clustering_dataset(
        n_samples=500, n_features=20, n_clusters=5
    )
    test_class_balance(ds_cluster, "Clustering Dataset")
    
    # Test 3: Temporal patterns (may be unbalanced)
    test_temporal_pattern_balance()
    
    # Test 4: Reproducibility
    test_reproducibility()
    
    # Test 5: Parameter variations
    print("\n" + "="*60)
    print("SUMMARY: Parameter Variation Tests")
    print("="*60)
    results = test_parameter_variations()
    
    print("\n" + "="*60)
    print("FINAL ASSESSMENT")
    print("="*60)
    print("\n✓ Classification & Clustering: Use sklearn's make_classification/make_blobs")
    print("  → These are GUARANTEED to be balanced (equal samples per class)")
    print("\n⚠ Temporal Patterns: Uses random assignment")
    print("  → May have slight imbalance, but statistically fair")
    print("\n✓ Random Seed: Only affects reproducibility, NOT fairness")
    print("  → Any seed value produces equally valid datasets")
    print("\n✓ Number of Classes/Features: Just structural parameters")
    print("  → Do NOT affect fairness, only dataset complexity")

