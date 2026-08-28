import os
import copy
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
from torchvision import datasets, transforms, models
from PIL import Image
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, precision_score, recall_score, silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import math


class DynamicWeightManager:
    def __init__(self, use_ema=False):
        self.c = cfg.NUM_CLASSES
        self.counts = np.array(cfg.CLASS_COUNTS, dtype=np.float64)
        
        # Cube-Root Inverse Frequency for early Representation Learning (3.0:1)
        # Sqrt (5.3x) and Linear (28x) amplify noise too much on 1-sample tail classes.
        self.w_rep = (self.counts.max() / self.counts) ** 0.33
        
        # Cube-Root for the base of the Dynamic Penalty (3.0:1)
        # Prevents massive gradient instability on 1-shot validation classes.
        self.w_pen_base = (self.counts.max() / self.counts) ** 0.33
        
        self.w_d = self._norm(self.w_rep)
        self.f_smooth = None
        self.initialized = False
        self.wd_history = []
        
    def _norm(self, w):
        return self.c * w / (w.sum() + 1e-9)
        
    def update(self, val_f1, val_ece):
        self.initialized = True
        
        if not hasattr(self, 'sigma2_f'):
            self.sigma2_f = np.ones(self.c) * 0.01
            self.f_smooth = np.ones(self.c) / self.c
            self.e_smooth = np.ones(self.c) * 0.05
            
        if getattr(self, 'use_ema', True):
            # Variance-weighted EMA smoothing (Critical for tiny val sets like Stanford Cars)
            inv = val_f1 - self.f_smooth
            self.sigma2_f = 0.9 * self.sigma2_f + 0.1 * (inv**2)
            adapt_lr = 0.1 / (0.1 + (self.sigma2_f / (self.sigma2_f.mean() + 1e-9)))
            self.f_smooth += adapt_lr * inv
            self.e_smooth = 0.9 * self.e_smooth + 0.1 * val_ece
        else:
            self.f_smooth = val_f1.copy()
            self.e_smooth = val_ece.copy()
        
        f1_max = self.f_smooth.max()
        if f1_max > 0:
            dynamic_multiplier = 1.0 + np.log((f1_max + 1e-9) / (self.f_smooth + 1e-9))
        else:
            dynamic_multiplier = np.ones_like(self.f_smooth)
            
        self.w_dynamic = self.w_pen_base * dynamic_multiplier
        
    def get_active_weights(self, gamma=0.0):
        if not self.initialized:
            return self._norm(self.w_rep)
        w_active = (1.0 - gamma) * self.w_rep + gamma * self.w_dynamic
        normed_w = self._norm(w_active)
        self.wd_history.append(normed_w.copy())
        return normed_w

class RLPenaltyScheduler:
    """
    Contextual Bandit for calibration penalty scheduling.
    State:  [F1_1..F1_K, ECE_1..ECE_K, epoch/max_epoch]
    Action: λ ∈ {0, 0.01, 0.03, 0.05, 0.1}
    Reward: ΔF1 (change in validation Macro-F1)
    """
    def __init__(self):
        self.actions = cfg.RL_ACTIONS
        self.n_actions = len(self.actions)
        self.state_dim = 2 * cfg.NUM_CLASSES + 1
        self.W = np.zeros((self.n_actions, self.state_dim))
        self.prev_state = None
        self.prev_action_idx = None
        self.prev_f1 = 0.0
        self.lambda_history = []
        
    def get_state(self, f1s, eces, epoch):
        return np.concatenate([f1s, eces, [epoch / cfg.EPOCHS]])
    
    def select_action(self, state):
        if np.random.rand() < cfg.RL_EPSILON:
            idx = np.random.randint(self.n_actions)
        else:
            q_values = self.W @ state
            idx = np.argmax(q_values)
        self.lambda_history.append(self.actions[idx])
        return idx, self.actions[idx]
    
    def update(self, state, f1):
        reward = f1 - self.prev_f1
        if self.prev_state is not None:
            q_next = np.max(self.W @ state)
            target = reward + cfg.RL_GAMMA * q_next
            q_prev = self.W[self.prev_action_idx] @ self.prev_state
            self.W[self.prev_action_idx] += cfg.RL_LR * (target - q_prev) * self.prev_state
        self.prev_f1 = f1

class HALOLoss(nn.Module):
    def __init__(self):
        super(HALOLoss, self).__init__()
        # We no longer hardcode weights here; they are fully managed by DynamicWeightManager
        
    def forward(self, logits, targets, w_active=None, gamma=0.0):
        if w_active is not None:
            # Use reduction='none' + manual multiply — confirmed best on Stanford Cars
            # Increased label_smoothing to 0.2 to prevent overconfidence on 1-shot tail classes
            ce = F.cross_entropy(logits, targets, reduction='none', label_smoothing=0.2)
            w = w_active[targets]
            return (ce * w).mean()
        else:
            return F.cross_entropy(logits, targets, label_smoothing=0.2)

