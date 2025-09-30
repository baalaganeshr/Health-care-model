"""Self-Adaptive Deep Neural Network (SA-DNN) with Gaussian Activation.

Implements SA-DNN with Gaussian activation function and utilities
for parameter flattening/unflattening for optimization algorithms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any


class GaussianActivation(nn.Module):
    """Gaussian activation function: φ(x) = exp(-α * x^2)."""
    
    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self.alpha * x.pow(2))


class SADNN(nn.Module):
    """Self-Adaptive Deep Neural Network with Gaussian activation.
    
    Args:
        input_dim: Input feature dimension
        hidden_sizes: List of hidden layer sizes
        gaussian_alpha: Alpha parameter for Gaussian activation
        dropout: Dropout probability (0 to disable)
    """
    
    def __init__(self, input_dim: int, hidden_sizes: List[int],
                 gaussian_alpha: float = 0.5, dropout: float = 0.0):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_sizes = hidden_sizes
        self.gaussian_alpha = gaussian_alpha
        self.dropout = dropout
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(GaussianActivation(gaussian_alpha))
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        # Output layer (single logit for binary classification)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Logits tensor of shape (batch_size, 1)
        """
        return self.network(x)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Predict class probabilities.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Probabilities tensor of shape (batch_size, 2) for [class_0, class_1]
        """
        with torch.no_grad():
            logits = self.forward(x)
            probs_class_1 = torch.sigmoid(logits)
            probs_class_0 = 1 - probs_class_1
            return torch.cat([probs_class_0, probs_class_1], dim=1)
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            'input_dim': self.input_dim,
            'hidden_sizes': self.hidden_sizes,
            'gaussian_alpha': self.gaussian_alpha,
            'dropout': self.dropout
        }


def flatten_params(model: nn.Module) -> torch.Tensor:
    """Flatten model parameters into 1D tensor.
    
    Args:
        model: PyTorch model
        
    Returns:
        1D tensor containing all model parameters
    """
    params = []
    for param in model.parameters():
        params.append(param.data.flatten())
    return torch.cat(params)


def unflatten_params(model: nn.Module, flat_params: torch.Tensor) -> None:
    """Unflatten 1D tensor back to model parameters.
    
    Args:
        model: PyTorch model to update
        flat_params: 1D tensor containing parameter values
    """
    idx = 0
    for param in model.parameters():
        param_size = param.numel()
        param.data = flat_params[idx:idx + param_size].reshape(param.shape)
        idx += param_size


def create_sadnn(input_dim: int, cfg: Dict[str, Any]) -> SADNN:
    """Create SA-DNN model from configuration.
    
    Args:
        input_dim: Input feature dimension
        cfg: Configuration dictionary
        
    Returns:
        Configured SA-DNN model
    """
    hidden_sizes = cfg.get('hidden_sizes', [64, 32])
    gaussian_alpha = cfg.get('gaussian_alpha', 0.5)
    dropout = cfg.get('dropout', 0.0)
    
    return SADNN(
        input_dim=input_dim,
        hidden_sizes=hidden_sizes,
        gaussian_alpha=gaussian_alpha,
        dropout=dropout
    )
