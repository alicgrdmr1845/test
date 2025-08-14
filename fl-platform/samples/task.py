"""
Sample FL Task - Simple Binary Classification Model
This file defines the model and training functions for federated learning
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Tuple, Dict, Any


# Model Definition
class SimpleNet(nn.Module):
    """Simple feedforward network for binary classification"""
    
    def __init__(self, input_dim=10, hidden_dim=20):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


def create_model() -> nn.Module:
    """Create and return the model instance"""
    return SimpleNet()


def get_weights(model: nn.Module) -> List[np.ndarray]:
    """Extract model weights as list of numpy arrays"""
    weights = []
    for param in model.parameters():
        weights.append(param.detach().cpu().numpy())
    return weights


def set_weights(model: nn.Module, weights: List[np.ndarray]) -> None:
    """Set model weights from list of numpy arrays"""
    for param, weight in zip(model.parameters(), weights):
        param.data = torch.tensor(weight, dtype=param.dtype)


def create_dataloader(split: str, config: Dict[str, Any]) -> DataLoader:
    """
    Create data loader for the specified split
    In production, this would load actual local data based on dataset_label
    For demo, returns dummy random data
    """
    # For demo: create dummy data
    num_samples = 500 if split == "train" else 100
    batch_size = config.get("batch_size", 32)
    
    # Random features and binary labels
    X = torch.randn(num_samples, 10)
    # Create some pattern in the data (not completely random)
    y = (X[:, 0] + X[:, 1] > 0).float().unsqueeze(1)
    
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"))


def fit(model: nn.Module, trainloader: DataLoader, config: Dict[str, Any]) -> Tuple[List[np.ndarray], int, Dict[str, float]]:
    """
    Train the model on local data
    
    Args:
        model: PyTorch model
        trainloader: Training data loader
        config: Training configuration
        
    Returns:
        Updated weights, number of examples, and metrics
    """
    # Training hyperparameters
    epochs = config.get("local_epochs", 5)
    learning_rate = config.get("learning_rate", 0.01)
    
    # Setup training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()
    
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    # Training loop
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_correct = 0
        epoch_samples = 0
        
        for batch_idx, (data, target) in enumerate(trainloader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            # Track metrics
            epoch_loss += loss.item()
            predictions = (output > 0.5).float()
            epoch_correct += (predictions == target).sum().item()
            epoch_samples += len(data)
        
        total_loss += epoch_loss
        total_correct += epoch_correct
        total_samples += epoch_samples
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss/len(trainloader):.4f}, "
              f"Accuracy: {epoch_correct/epoch_samples:.4f}")
    
    # Calculate final metrics
    avg_loss = total_loss / (epochs * len(trainloader))
    accuracy = total_correct / total_samples
    
    metrics = {
        "loss": avg_loss,
        "accuracy": accuracy,
        "epochs_trained": epochs
    }
    
    # Return updated weights and metrics
    weights = get_weights(model)
    num_examples = len(trainloader.dataset)
    
    return weights, num_examples, metrics


def evaluate(model: nn.Module, testloader: DataLoader, config: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    """
    Evaluate the model on test data
    
    Args:
        model: PyTorch model
        testloader: Test data loader
        config: Evaluation configuration
        
    Returns:
        Loss and metrics dictionary
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    criterion = nn.BCELoss()
    
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            predictions = (output > 0.5).float()
            total_correct += (predictions == target).sum().item()
            total_samples += len(data)
    
    avg_loss = total_loss / len(testloader)
    accuracy = total_correct / total_samples
    
    metrics = {
        "test_loss": avg_loss,
        "test_accuracy": accuracy
    }
    
    return avg_loss, metrics


# Optional: Dataset mapping for production use
DATASET_MAPPING = {
    "default": "/data/default",
    "medical": "/data/medical",
    "financial": "/data/financial",
    # Client would map these to actual local paths
}


if __name__ == "__main__":
    # Test the task locally
    print("Testing task.py locally...")
    
    # Create model
    model = create_model()
    print(f"Model created: {model}")
    
    # Create dummy data
    config = {"batch_size": 32, "local_epochs": 2}
    trainloader = create_dataloader("train", config)
    testloader = create_dataloader("test", config)
    
    # Test training
    weights, num_examples, metrics = fit(model, trainloader, config)
    print(f"Training completed: {num_examples} examples, metrics: {metrics}")
    
    # Test evaluation
    test_loss, test_metrics = evaluate(model, testloader, config)
    print(f"Evaluation completed: loss={test_loss:.4f}, metrics: {test_metrics}")