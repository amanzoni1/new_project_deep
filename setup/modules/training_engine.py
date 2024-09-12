"""
Generalized functions for training and evaluating a PyTorch model for classification and regression.
"""

from typing import Dict, List, Tuple, Optional
import torch
from tqdm.auto import tqdm

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               criterion: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device,
               task_type: str = "classification") -> Tuple[float, Optional[float]]:
    """Performs one training step, updating model weights and returning loss and accuracy (if classification)."""
    model.train()
    train_loss, correct, total = 0, 0, 0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()

        y_pred = model(X)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        if task_type == "classification":
            y_pred_class = torch.argmax(y_pred, dim=1)
            correct += (y_pred_class == y).sum().item()
        total += y.size(0)

    train_loss /= len(dataloader)
    train_acc = correct / total if task_type == "classification" else None

    return train_loss, train_acc

def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              criterion: torch.nn.Module,
              device: torch.device,
              task_type: str = "classification") -> Tuple[float, Optional[float]]:
    """Performs one evaluation step, returning loss and accuracy (if classification)."""
    model.eval()
    test_loss, correct, total = 0, 0, 0

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)

            test_loss += criterion(y_pred, y).item()
            if task_type == "classification":
                y_pred_class = torch.argmax(y_pred, dim=1)
                correct += (y_pred_class == y).sum().item()
            total += y.size(0)

    test_loss /= len(dataloader)
    test_acc = correct / total if task_type == "classification" else None

    return test_loss, test_acc

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          criterion: torch.nn.Module,
          epochs: int,
          device: torch.device,
          task_type: str = "classification") -> Dict[str, List[Optional[float]]]:
    """Runs full training and evaluation loop, returning metrics for each epoch."""
    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model, train_dataloader, criterion, optimizer, device, task_type)
        test_loss, test_acc = test_step(model, test_dataloader, criterion, device, task_type)

        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, "
              f"Train Acc: {train_acc if train_acc is not None else 'N/A'}, "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc if test_acc is not None else 'N/A'}")

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    return results
