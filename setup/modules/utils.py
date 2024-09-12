"""
Utility functions for PyTorch model training, saving, and additional tasks.
"""

from pathlib import Path
import torch

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    """Saves a PyTorch model's state_dict to the specified directory.

    Args:
    model: A PyTorch model to save.
    target_dir: Directory to save the model.
    model_name: Name of the saved model file, should end with '.pth' or '.pt'.
    """
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model's state_dict
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)


def load_model(model: torch.nn.Module,
               model_path: str,
               device: torch.device = torch.device("cpu")) -> torch.nn.Module:
    """Loads a PyTorch model's state_dict from a file.

    Args:
    model: The PyTorch model instance where the state_dict will be loaded.
    model_path: Path to the model state_dict file.
    device: The device to load the model on.

    Returns:
    The model with loaded parameters.
    """
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"[INFO] Loaded model from: {model_path}")
    return model


def save_training_results(results: dict,
                          target_dir: str,
                          filename: str = "training_results.json"):
    """Saves the training metrics (loss/accuracy) into a JSON file.

    Args:
    results: A dictionary containing training/test loss and accuracy.
    target_dir: Directory to save the results file.
    filename: Name of the JSON file to save results in.
    """
    import json
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    save_path = target_dir_path / filename
    with open(save_path, "w") as f:
        json.dump(results, f)

    print(f"[INFO] Saved training results to: {save_path}")
