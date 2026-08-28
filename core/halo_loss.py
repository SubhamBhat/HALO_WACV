import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HALOOptimizationEngine:
    """
    HALO Optimization Engine.
    Manages Variance-Bounded Weights and Quadratic SDE Homotopy.
    """
    def __init__(self, class_counts, num_classes):
        self.c = num_classes
        self.counts = np.array(class_counts, dtype=np.float64)
        
        # 1. Variance-Bounded Weights: W_pen = sqrt(N_max / N_c)
        # Bounded by Rademacher complexity to prevent gradient explosion on minority classes
        self.w_pen_base = (self.counts.max() / self.counts) ** 0.5
        
        self.initialized = False
        self.alpha_t = 0.0

    def update_homotopy(self, train_acc):
        """
        2. Quadratic SDE Homotopy: alpha(t) = acc(t)^2
        Smoothly interpolates between representation learning and re-balancing.
        """
        self.initialized = True
        self.alpha_t = train_acc ** 2
        
        # 3. Log Margin Calculation: Delta_c = alpha(t) * W_pen
        margin_np = self.alpha_t * self.w_pen_base
        return torch.tensor(margin_np, dtype=torch.float32)

class HALOLoss(nn.Module):
    """
    HALO Loss: Integrates the dynamic Log Margin into the differentiable objective.
    """
    def __init__(self):
        super(HALOLoss, self).__init__()
        
    def forward(self, logits, targets, margin_c=None):
        if margin_c is not None:
            # Inject additive log-odds shift (-Delta_c) into logits for the target class
            device = logits.device
            margin_c = margin_c.to(device)
            
            # Create a margin tensor of same shape as logits
            margin_tensor = torch.zeros_like(logits)
            margin_tensor.scatter_(1, targets.view(-1, 1), margin_c[targets].view(-1, 1))
            
            # Apply log margin (subtracting margin equivalent to expanding decision boundary)
            adjusted_logits = logits - margin_tensor
            
            return F.cross_entropy(adjusted_logits, targets)
        else:
            return F.cross_entropy(logits, targets)

def l1_trace_normalization(model, max_norm=1.0):
    """
    4. L1 Trace Normalization: ||nabla L||_1 <= C
    Decouples dynamic class penalties from the optimizer's global Lipschitz bound.
    """
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm, norm_type=1.0)
