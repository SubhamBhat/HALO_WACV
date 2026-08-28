import torch
import numpy as np
from sklearn.metrics import f1_score

def calibrate_tau_grid_search(val_logits, val_targets, class_counts, tau_range=(0.0, 3.0), step=0.01):
    """
    Post-hoc Calibration: tau* = argmax_{tau} Macro-F1(val)
    Adjusts logits using the class priors: f~_c = f_c - tau * log(pi_c)
    
    Args:
        val_logits (np.ndarray or torch.Tensor): Raw validation logits.
        val_targets (np.ndarray or torch.Tensor): Validation ground truth labels.
        class_counts (list or np.ndarray): Number of training samples per class.
        tau_range (tuple): (min_tau, max_tau) search space.
        step (float): Grid search step size.
        
    Returns:
        float: Optimal tau (tau*)
        float: Best Macro-F1 score achieved on the validation set.
    """
    if torch.is_tensor(val_logits):
        val_logits = val_logits.cpu().numpy()
    if torch.is_tensor(val_targets):
        val_targets = val_targets.cpu().numpy()
        
    counts = np.array(class_counts, dtype=np.float64)
    priors = counts / counts.sum()
    log_priors = np.log(priors + 1e-9)
    
    best_tau = 0.0
    best_f1 = -1.0
    
    for tau in np.arange(tau_range[0], tau_range[1] + step, step):
        # Apply shift: f~_c = f_c - tau * log(pi_c)
        shifted_logits = val_logits - tau * log_priors
        
        preds = np.argmax(shifted_logits, axis=1)
        f1 = f1_score(val_targets, preds, average='macro')
        
        if f1 > best_f1:
            best_f1 = f1
            best_tau = tau
            
    return best_tau, best_f1

def apply_calibration(logits, class_counts, tau_star):
    """
    Applies the optimal tau* to inference logits to get the final predictions.
    """
    counts = np.array(class_counts, dtype=np.float64)
    priors = counts / counts.sum()
    log_priors = np.log(priors + 1e-9)
    
    if torch.is_tensor(logits):
        device = logits.device
        log_priors = torch.tensor(log_priors, dtype=torch.float32, device=device)
        return logits - tau_star * log_priors
    else:
        return logits - tau_star * log_priors
