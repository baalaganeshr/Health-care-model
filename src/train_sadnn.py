"""SA-DNN Training with IWOA Optimization.

Trains Self-Adaptive Deep Neural Network with Gaussian activation
using Improved Whale Optimization Algorithm for weight optimization.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Tuple

import yaml
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from .dataio import load_csv, clean
from .nets import create_sadnn
from .opt import optimize_weights


def prepare_data(config: Dict[str, Any], seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any, Any]:
    """Prepare data for SA-DNN training."""
    # Load and clean data
    data_config = config.get('data', {})
    csv_path = data_config.get('csv_path', 'data/heart.csv')
    target_col = data_config.get('target', 'target')
    
    df = load_csv(csv_path)
    df_clean = clean(df)
    
    # Separate features and target
    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col]
    
    # Split data
    split_config = data_config.get('split', {})
    test_size = split_config.get('test_size', 0.3)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply PCA if configured
    pca = None
    features_config = config.get('features', {})
    if features_config.get('pca', False):
        n_components = features_config.get('pca_components', 0.95)
        pca = PCA(n_components=n_components, random_state=seed)
        X_train_scaled = pca.fit_transform(X_train_scaled)
        X_test_scaled = pca.transform(X_test_scaled)
    
    # Convert to tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)
    
    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, scaler, pca


def create_fitness_function(X_val: torch.Tensor, y_val: torch.Tensor, 
                           batch_size: int = 32) -> callable:
    """Create fitness evaluation function for IWOA."""
    def fitness_fn(model: nn.Module) -> float:
        model.eval()
        total_loss = 0.0
        n_batches = 0
        
        with torch.no_grad():
            # Mini-batch evaluation for efficiency
            for i in range(0, len(X_val), batch_size):
                batch_X = X_val[i:i+batch_size]
                batch_y = y_val[i:i+batch_size]
                
                outputs = model(batch_X)
                loss = nn.BCEWithLogitsLoss()(outputs, batch_y)
                total_loss += loss.item()
                n_batches += 1
        
        return total_loss / max(1, n_batches)
    
    return fitness_fn


def train_with_adam(model: nn.Module, X_train: torch.Tensor, y_train: torch.Tensor,
                   X_val: torch.Tensor, y_val: torch.Tensor, config: Dict[str, Any]) -> Dict[str, Any]:
    """Train SA-DNN with Adam optimizer (fallback)."""
    lr = config.get('lr', 0.0001)
    epochs = config.get('epochs', 30)
    batch_size = config.get('batch_size', 20)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        n_batches = 0
        
        # Mini-batch training
        indices = torch.randperm(len(X_train))
        for i in range(0, len(X_train), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_X = X_train[batch_indices]
            batch_y = y_train[batch_indices]
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            n_batches += 1
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val).item()
        
        avg_train_loss = train_loss / max(1, n_batches)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    return history


def evaluate_model(model: nn.Module, X_test: torch.Tensor, y_test: torch.Tensor) -> Dict[str, float]:
    """Evaluate trained model."""
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        probabilities = torch.sigmoid(outputs)
        predictions = (probabilities > 0.5).float()
        
        # Convert to numpy for sklearn metrics
        y_true = y_test.cpu().numpy().flatten()
        y_pred = predictions.cpu().numpy().flatten()
        y_prob = probabilities.cpu().numpy().flatten()
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_prob)
        }
    
    return metrics


def run(config_path: Path, seed: int = 42) -> Dict[str, Any]:
    """Run SA-DNN training with IWOA optimization.
    
    Args:
        config_path: Path to configuration file
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with training results and artifact paths
    """
    print("🧠 Starting SA-DNN Training with IWOA Optimization")
    print("=" * 60)
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    sadnn_config = config.get('sadnn', {})
    
    # Prepare data
    print("📊 Preparing data...")
    X_train, X_test, y_train, y_test, scaler, pca = prepare_data(config, seed)
    
    # Create validation split from training data
    val_size = 0.2
    n_val = int(len(X_train) * val_size)
    indices = torch.randperm(len(X_train))
    
    X_val = X_train[indices[:n_val]]
    y_val = y_train[indices[:n_val]]
    X_train_sub = X_train[indices[n_val:]]
    y_train_sub = y_train[indices[n_val:]]
    
    print(f"  📈 Training samples: {len(X_train_sub)}")
    print(f"  📊 Validation samples: {len(X_val)}")
    print(f"  🎯 Test samples: {len(X_test)}")
    print(f"  🔢 Input features: {X_train.shape[1]}")
    
    # Create SA-DNN model
    print("🏗️ Building SA-DNN model...")
    model = create_sadnn(input_dim=X_train.shape[1], cfg=sadnn_config)
    print(f"  🧪 Model architecture: {model}")
    
    # Check IWOA configuration
    iwoa_config = sadnn_config.get('iwoa', {})
    use_iwoa = iwoa_config.get('enabled', True)
    
    artifact_paths = {}
    training_history = {}
    
    if use_iwoa:
        print("🐋 Training with IWOA weight optimization...")
        
        # Create fitness function
        fitness_fn = create_fitness_function(X_val, y_val, sadnn_config.get('batch_size', 20))
        
        # Run IWOA optimization
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"  💻 Using device: {device}")
            
            best_state_dict, iwoa_history = optimize_weights(
                model=model,
                fit_eval_fn=fitness_fn,
                cfg=sadnn_config,
                device=device
            )
            
            # Load best weights
            model.load_state_dict(best_state_dict)
            training_history = {'iwoa_fitness': iwoa_history}
            
            model_filename = "sa_dnn_iwoa.pt"
            
        except Exception as e:
            print(f"  ⚠️ IWOA optimization failed: {e}")
            print("  🔄 Falling back to Adam optimizer...")
            use_iwoa = False
    
    if not use_iwoa:
        print("🚀 Training with Adam optimizer...")
        training_history = train_with_adam(model, X_train_sub, y_train_sub, X_val, y_val, sadnn_config)
        model_filename = "sa_dnn.pt"
    
    # Evaluate model
    print("📊 Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test)
    
    print("\n🎯 SA-DNN Performance:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    # Save model
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / model_filename
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': model.get_config(),
        'scaler': scaler,
        'pca': pca,
        'metrics': metrics,
        'training_history': training_history,
        'optimizer_used': 'IWOA' if use_iwoa else 'Adam'
    }, model_path)
    artifact_paths['model'] = str(model_path)
    
    # Save metrics
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)
    
    metrics_path = artifacts_dir / "metrics_sadnn.json"
    with open(metrics_path, 'w') as f:
        json.dump({
            'model_type': 'SA-DNN',
            'optimizer': 'IWOA' if use_iwoa else 'Adam',
            'metrics': metrics,
            'training_config': sadnn_config,
            'data_shapes': {
                'train': X_train_sub.shape,
                'val': X_val.shape,
                'test': X_test.shape
            }
        }, f, indent=2)
    artifact_paths['metrics'] = str(metrics_path)
    
    print(f"\n✅ SA-DNN training completed!")
    print(f"📁 Model saved: {model_path}")
    print(f"📊 Metrics saved: {metrics_path}")
    
    return {
        'model_type': 'SA-DNN',
        'optimizer_used': 'IWOA' if use_iwoa else 'Adam',
        'metrics': metrics,
        'artifact_paths': artifact_paths,
        'model_path': str(model_path)
    }
