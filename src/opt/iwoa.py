"""Improved Whale Optimization Algorithm (IWOA) for Neural Network Weight Optimization.

Implements the Improved Whale Optimization Algorithm with opposition-based learning
and adaptive parameters for optimizing deep neural network weights.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Callable, Dict, List, Tuple, Any
from dataclasses import dataclass
import copy


@dataclass
class IWOAConfig:
    """Configuration for IWOA algorithm."""
    pop_size: int = 20
    generations: int = 50
    a_decay: float = 0.98  # Decay factor for 'a' parameter
    spiral_prob: float = 0.5  # Probability of spiral behavior
    clamp: float = 1.0  # Weight clamping range [-clamp, +clamp]
    patience: int = 7  # Early stopping patience
    opposition_prob: float = 0.3  # Probability of opposition-based learning


class IWOAOptimizer:
    """Improved Whale Optimization Algorithm for neural network optimization."""
    
    def __init__(self, config: IWOAConfig):
        self.config = config
        self.best_whale = None
        self.best_fitness = float('inf')
        self.fitness_history = []
        self.population = []
        self.fitnesses = []
        
    def _flatten_params(self, model: nn.Module) -> torch.Tensor:
        """Flatten model parameters into 1D tensor."""
        params = []
        for param in model.parameters():
            params.append(param.data.flatten())
        return torch.cat(params)
    
    def _unflatten_params(self, model: nn.Module, flat_params: torch.Tensor) -> None:
        """Unflatten 1D tensor back to model parameters."""
        idx = 0
        for param in model.parameters():
            param_size = param.numel()
            param.data = flat_params[idx:idx + param_size].reshape(param.shape)
            idx += param_size
    
    def _clamp_weights(self, weights: np.ndarray) -> np.ndarray:
        """Clamp weights within specified range."""
        return np.clip(weights, -self.config.clamp, self.config.clamp)
    
    def _opposition_learning(self, whale: np.ndarray, bounds: Tuple[float, float]) -> np.ndarray:
        """Apply opposition-based learning."""
        min_bound, max_bound = bounds
        return min_bound + max_bound - whale
    
    def _initialize_population(self, dim: int) -> List[np.ndarray]:
        """Initialize whale population with Gaussian distribution."""
        population = []
        sigma = 0.1  # Standard deviation for initialization
        
        for _ in range(self.config.pop_size):
            whale = np.random.normal(0, sigma, dim)
            whale = self._clamp_weights(whale)
            population.append(whale)
            
            # Apply opposition-based learning with some probability
            if np.random.random() < self.config.opposition_prob:
                bounds = (-self.config.clamp, self.config.clamp)
                opposite_whale = self._opposition_learning(whale, bounds)
                opposite_whale = self._clamp_weights(opposite_whale)
                population.append(opposite_whale)
        
        # Trim population to desired size
        return population[:self.config.pop_size]
    
    def _update_whale_position(self, whale: np.ndarray, best_whale: np.ndarray,
                             a: float, generation: int, dim: int) -> np.ndarray:
        """Update whale position using WOA equations."""
        # Linearly decrease A from 2 to 0
        A = 2 * a * np.random.random() - a
        # Random values for C
        C = 2 * np.random.random()
        
        # Choose behavior based on |A|
        if np.abs(A) < 1:
            # Encircling prey (exploitation)
            D = np.abs(C * best_whale - whale)
            new_whale = best_whale - A * D
        else:
            # Search for prey (exploration)
            # Select random whale
            rand_idx = np.random.randint(0, len(self.population))
            X_rand = self.population[rand_idx]
            D = np.abs(C * X_rand - whale)
            new_whale = X_rand - A * D
        
        # Spiral behavior with probability
        if np.random.random() < self.config.spiral_prob:
            # Bubble-net attacking method (spiral)
            b = 1  # Spiral shape constant
            l = np.random.uniform(-1, 1)  # Random number in [-1, 1]
            
            D_spiral = np.abs(best_whale - whale)
            spiral_update = D_spiral * np.exp(b * l) * np.cos(2 * np.pi * l) + best_whale
            
            # Blend spiral and regular update
            new_whale = 0.5 * new_whale + 0.5 * spiral_update
        
        return self._clamp_weights(new_whale)
    
    def optimize_weights(self, model: nn.Module, fit_eval_fn: Callable,
                        device: torch.device = None) -> Tuple[Dict[str, torch.Tensor], List[float]]:
        """Optimize neural network weights using IWOA.
        
        Args:
            model: PyTorch neural network model
            fit_eval_fn: Function that evaluates fitness given model
            device: Device to run optimization on
            
        Returns:
            Tuple of (best_state_dict, fitness_history)
        """
        if device is None:
            device = torch.device('cpu')
        
        model = model.to(device)
        
        # Get parameter dimensions
        sample_params = self._flatten_params(model)
        dim = len(sample_params)
        
        print(f"Optimizing {dim} parameters with IWOA...")
        
        # Initialize population
        self.population = self._initialize_population(dim)
        self.fitnesses = []
        self.best_whale = None
        self.best_fitness = float('inf')
        self.fitness_history = []
        
        # Evaluate initial population
        for whale in self.population:
            # Set model weights
            whale_tensor = torch.tensor(whale, dtype=torch.float32, device=device)
            self._unflatten_params(model, whale_tensor)
            
            # Evaluate fitness
            fitness = fit_eval_fn(model)
            self.fitnesses.append(fitness)
            
            # Update best solution
            if fitness < self.best_fitness:
                self.best_fitness = fitness
                self.best_whale = whale.copy()
        
        self.fitness_history.append(self.best_fitness)
        
        # Early stopping tracking
        best_fitness_tracker = self.best_fitness
        patience_counter = 0
        
        # Main optimization loop
        for generation in range(self.config.generations):
            # Update 'a' parameter (linearly decreases from 2 to 0)
            a = 2 * (1 - generation / self.config.generations) * self.config.a_decay
            
            # Update each whale
            new_population = []
            new_fitnesses = []
            
            for i, whale in enumerate(self.population):
                # Update whale position
                new_whale = self._update_whale_position(
                    whale, self.best_whale, a, generation, dim
                )
                
                # Evaluate new position
                whale_tensor = torch.tensor(new_whale, dtype=torch.float32, device=device)
                self._unflatten_params(model, whale_tensor)
                fitness = fit_eval_fn(model)
                
                # Keep better solution
                if fitness < self.fitnesses[i]:
                    new_population.append(new_whale)
                    new_fitnesses.append(fitness)
                else:
                    new_population.append(whale)
                    new_fitnesses.append(self.fitnesses[i])
                
                # Update global best
                if fitness < self.best_fitness:
                    self.best_fitness = fitness
                    self.best_whale = new_whale.copy()
            
            self.population = new_population
            self.fitnesses = new_fitnesses
            self.fitness_history.append(self.best_fitness)
            
            # Early stopping check
            if self.best_fitness < best_fitness_tracker - 1e-6:
                best_fitness_tracker = self.best_fitness
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= self.config.patience:
                print(f"Early stopping at generation {generation + 1}")
                break
            
            if (generation + 1) % 10 == 0:
                print(f"Generation {generation + 1}: Best fitness = {self.best_fitness:.6f}")
        
        # Set model to best weights
        best_tensor = torch.tensor(self.best_whale, dtype=torch.float32, device=device)
        self._unflatten_params(model, best_tensor)
        
        # Return best state dict
        best_state_dict = copy.deepcopy(model.state_dict())
        
        print(f"IWOA optimization completed. Best fitness: {self.best_fitness:.6f}")
        
        return best_state_dict, self.fitness_history


def optimize_weights(model: nn.Module, fit_eval_fn: Callable,
                    cfg: Dict[str, Any], device: torch.device = None) -> Tuple[Dict[str, torch.Tensor], List[float]]:
    """Convenience function for IWOA weight optimization.
    
    Args:
        model: PyTorch neural network model
        fit_eval_fn: Function that evaluates fitness (lower is better)
        cfg: Configuration dictionary containing IWOA parameters
        device: Device to run optimization on
        
    Returns:
        Tuple of (best_state_dict, fitness_history)
    """
    iwoa_cfg = cfg.get('iwoa', {})
    config = IWOAConfig(
        pop_size=iwoa_cfg.get('pop_size', 20),
        generations=iwoa_cfg.get('generations', 50),
        a_decay=iwoa_cfg.get('a_decay', 0.98),
        spiral_prob=iwoa_cfg.get('spiral_prob', 0.5),
        clamp=iwoa_cfg.get('clamp', 1.0),
        patience=iwoa_cfg.get('patience', 7),
        opposition_prob=iwoa_cfg.get('opposition_prob', 0.3)
    )
    
    optimizer = IWOAOptimizer(config)
    return optimizer.optimize_weights(model, fit_eval_fn, device)
