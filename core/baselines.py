import torch
import torch.nn as nn
import torch.nn.functional as F

class PublishedDCALoss(nn.Module):
    """
    Published DCAL (Hebbalaguppe et al., CVPR 2022)
    Combines ArcFace margin with MDCA (Multi-class Difference in Confidence and Accuracy) calibration loss.
    """
    def __init__(self, m=0.3, s=30.0, lambda_mdca=0.1):
        super().__init__()
        self.m = m
        self.s = s
        self.lambda_mdca = lambda_mdca
        
    def forward(self, logits, targets, w_active=None):
        # 1. ArcFace Angular Margin
        norms = torch.norm(logits, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        cosine = logits / norms
        
        index = torch.zeros_like(cosine, dtype=torch.bool)
        index.scatter_(1, targets.data.view(-1, 1), 1)
        cosine_m = cosine - self.m
        
        output = torch.where(index, cosine_m, cosine) * self.s
        
        ce_loss = F.cross_entropy(output, targets, weight=w_active)
        
        # 2. MDCA Calibration Penalty
        probs = F.softmax(logits, dim=1)
        conf = probs.max(dim=1)[0]
        preds = probs.argmax(dim=1)
        correct = (preds == targets).float()
        
        mdca_loss = 0.0
        unique_classes = torch.unique(targets)
        for c in unique_classes:
            class_mask = (targets == c)
            if class_mask.sum() > 0:
                class_conf = conf[class_mask].mean()
                class_acc = correct[class_mask].mean()
                # Difference between average confidence and accuracy per class
                mdca_loss += torch.abs(class_conf - class_acc)
                
        mdca_loss = mdca_loss / len(unique_classes)
        
        return ce_loss + self.lambda_mdca * mdca_loss
