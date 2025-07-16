import torch
import numpy as np
import random
import math
import warnings
from torch.optim.lr_scheduler import _LRScheduler

def calculate_transformed_ratio_sigma(loss_history, volatility_window_N, weight_const, min_loss_val=1e-8):
    data = np.maximum(np.array(loss_history), min_loss_val)
    required_data = volatility_window_N + 1
    if len(data) < required_data:
        return 1.0

    log_returns = np.log(data[1:] / data[:-1])
    sigma_all_val = np.std(log_returns, ddof=1)
    recent_log_returns_N = log_returns[-volatility_window_N:]
    sigma_n_val = np.std(recent_log_returns_N, ddof=1)

    if sigma_n_val == 0 or math.isnan(sigma_n_val) or math.isinf(sigma_n_val) or \
       math.isnan(sigma_all_val) or math.isinf(sigma_all_val):
        return 1.0

    ratio_sigma_val = sigma_all_val / sigma_n_val
    
    signed_deviation = ratio_sigma_val - 1.0
    abs_deviation = np.abs(signed_deviation)
    scaled_abs_deviation = abs_deviation * weight_const
    
    log_transformed_deviation = np.log1p(scaled_abs_deviation) 
    
    final_signed_log_deviation = np.copysign(log_transformed_deviation, signed_deviation)
    
    final_multiplier = 1.0 + final_signed_log_deviation
    
    if math.isnan(final_multiplier) or math.isinf(final_multiplier):
        return 1.0
        
    return final_multiplier

class VolSched(_LRScheduler):
    def __init__(self, optimizer,
                 volatility_window_N,
                 weight_const,
                 T_max,
                 eta_min = 0,
                 max_loss_history_len=None,
                 last_epoch=-1, verbose=False,
                 warmup_epochs=0, warmup_lr_init=0):
        
        if volatility_window_N < 2:
            raise ValueError("volatility_window_N must be at least 2.")
        if warmup_epochs >= T_max:
            raise ValueError("warmup_epochs must be smaller than T_max.")

        self.T_max = T_max
        self.volatility_window_N = volatility_window_N
        self.scheduler_update_interval = volatility_window_N
        self.weight_const = weight_const
        self.loss_history = []
        if max_loss_history_len is None:
            self.max_loss_history_len = volatility_window_N * 10
        else:
            self.max_loss_history_len = max_loss_history_len
        self.eta_min = eta_min
        self.warmup_epochs = warmup_epochs
        self.warmup_lr_init = warmup_lr_init
        
        super(VolSched, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)
        
        if self.last_epoch < self.warmup_epochs:
            if self.warmup_epochs > 1:
                warmup_factor = self.last_epoch / (self.warmup_epochs - 1)
                return [self.warmup_lr_init * base_lr + base_lr * (1 - self.warmup_lr_init) * warmup_factor
                        for base_lr in self.base_lrs]
            elif self.warmup_epochs == 1:
                return self.base_lrs
            return [group['lr'] for group in self.optimizer.param_groups]

        effective_epoch = self.last_epoch - self.warmup_epochs
        effective_T_max = self.T_max - self.warmup_epochs

        if effective_epoch == 0:
            return self.base_lrs
            
        if effective_epoch % self.scheduler_update_interval != 0:
            return [group['lr'] for group in self.optimizer.param_groups]
        
        cosmul = (1 + math.cos(math.pi * effective_epoch / effective_T_max)) / (1 + math.cos(math.pi * (effective_epoch - self.scheduler_update_interval) / effective_T_max))
        multiplier = calculate_transformed_ratio_sigma(
            loss_history=self.loss_history,
            volatility_window_N=self.volatility_window_N,
            weight_const=self.weight_const
        ) * cosmul
        if self.verbose:
            print(f"LR Scheduler: Update at end of step {self.last_epoch}, "
                  f"Old LRs: {[g['lr'] for g in self.optimizer.param_groups]}, "
                  f"New LRs: {[max(self.eta_min, g['lr'] * multiplier) for g in self.optimizer.param_groups]}, "
                  f"Cosine: {cosmul:.4f}, "
                  f"Calculated LR Multiplier: {multiplier:.4f}")
        
        return [max(self.eta_min, g['lr'] * multiplier) for g in self.optimizer.param_groups]

    def step(self, loss_value=None, epoch=None):
        if loss_value is not None:
            loss_val_float = float(loss_value)
            if math.isnan(loss_val_float) or math.isinf(loss_val_float):
                warnings.warn(f"NaN or Inf loss_value ({loss_value}) passed to scheduler.step(). Skipping update for this loss.")
            else:
                self.loss_history.append(loss_val_float)
            
            if self.max_loss_history_len is not None and len(self.loss_history) > self.max_loss_history_len:
                num_to_remove = len(self.loss_history) - self.max_loss_history_len
                self.loss_history = self.loss_history[num_to_remove:]
        
        super(VolSched, self).step(epoch)



#you probably shouldnt use this, but this is what i used for the resnet tests
class VolSchedNoWarmup(_LRScheduler):
    def __init__(self, optimizer,
                 volatility_window_N,
                 weight_const,
                 T_max,
                 eta_min = 0,
                 max_loss_history_len=None,
                 last_epoch=-1, verbose=False):
        
        if volatility_window_N < 2:
            raise ValueError("volatility_window_N must be at least 2.")
        self.T_max = T_max
        self.volatility_window_N = volatility_window_N
        self.scheduler_update_interval = volatility_window_N
        self.weight_const = weight_const
        self.loss_history = []
        if max_loss_history_len is None:
            self.max_loss_history_len = volatility_window_N * 10
        else:
            self.max_loss_history_len = max_loss_history_len
        self.eta_min = eta_min
        
        super(VolSchedNoWarmup, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)
        
        if self.last_epoch == 0:
            return [group['lr'] for group in self.optimizer.param_groups]
            
        if self.last_epoch % self.scheduler_update_interval != 0:
            return [group['lr'] for group in self.optimizer.param_groups]
        
        cosmul = (1 + math.cos(math.pi * (self.last_epoch) / self.T_max)) / (1 + math.cos(math.pi * (self.last_epoch - self.scheduler_update_interval) / self.T_max))
        multiplier = calculate_transformed_ratio_sigma(
            loss_history=self.loss_history,
            volatility_window_N=self.volatility_window_N,
            weight_const=self.weight_const
        ) * cosmul
        if self.verbose:
            print(f"LR Scheduler: Update at end of step {self.last_epoch}, "
                  f"Old LRs: {[g['lr'] for g in self.optimizer.param_groups]}, "
                  f"New LRs: {[max(self.eta_min, g['lr'] * multiplier) for g in self.optimizer.param_groups]}, "
                  f"Cosine: {cosmul:.4f}, "
                  f"Calculated LR Multiplier: {multiplier:.4f}")
        
        return [max(self.eta_min, g['lr'] * multiplier) for g in self.optimizer.param_groups]

    def step(self, loss_value=None, epoch=None):
        if loss_value is not None:
            loss_val_float = float(loss_value)
            if math.isnan(loss_val_float) or math.isinf(loss_val_float):
                warnings.warn(f"NaN or Inf loss_value ({loss_value}) passed to scheduler.step(). Skipping update for this loss.")
            else:
                self.loss_history.append(loss_val_float)
            
            if self.max_loss_history_len is not None and len(self.loss_history) > self.max_loss_history_len:
                num_to_remove = len(self.loss_history) - self.max_loss_history_len
                self.loss_history = self.loss_history[num_to_remove:]
        
        super(VolSchedNoWarmup, self).step(epoch)