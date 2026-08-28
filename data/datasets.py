import numpy as np
import torch
from torch.utils.data import Subset

def get_exponential_decay_counts(num_classes, max_samples, imb_factor):
    """
    Computes the sample counts for each class following an exponential decay distribution.
    """
    class_counts = []
    for c in range(num_classes):
        # N_c = N_max * (imb_factor ^ (c / (C - 1)))
        count = max_samples * (imb_factor ** (c / (num_classes - 1.0)))
        class_counts.append(max(1, int(count))) # Ensure at least 1 sample (extreme long tail)
    return class_counts

def create_long_tail_dataset(dataset, num_classes, max_samples, imb_factor):
    """
    Subsamples a balanced dataset into a long-tailed dataset using exponential decay.
    """
    target_counts = get_exponential_decay_counts(num_classes, max_samples, imb_factor)
    
    # Extract labels based on dataset type
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    elif hasattr(dataset, 'labels'):
        labels = np.array(dataset.labels)
    else:
        labels = np.array([y for _, y in dataset])
        
    class_indices = [np.where(labels == i)[0] for i in range(num_classes)]
    
    subset_indices = []
    actual_counts = []
    
    for i in range(num_classes):
        # If dataset doesn't have enough samples for the head class, clip it
        available = len(class_indices[i])
        n_keep = min(target_counts[i], available)
        
        np.random.seed(42) # Deterministic subsampling
        selected = np.random.choice(class_indices[i], n_keep, replace=False)
        subset_indices.extend(selected)
        actual_counts.append(n_keep)
        
    lt_dataset = Subset(dataset, subset_indices)
    return lt_dataset, actual_counts

# Example generic wrapper
def load_dataset(dataset_name, data_dir, image_size=224):
    """
    Returns (train_dataset, val_dataset, class_counts, num_classes)
    Implement specific torchvision/custom loading logic here.
    For reproducibility, datasets should be loaded using torchvision standard loaders
    and passed through `create_long_tail_dataset` with an imb_factor of 1/50 (0.02).
    """
    pass
