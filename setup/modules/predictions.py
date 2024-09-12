"""
Utility functions to make predictions and plot results for PyTorch models.

Reference: https://www.learnpytorch.io/06_pytorch_transfer_learning/#6-make-predictions-on-images-from-the-test-set
"""
import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from PIL import Image

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

def predict_and_plot(
    model: torch.nn.Module,
    class_names: List[str],
    image_path: str,
    image_size: Tuple[int, int] = (224, 224),
    transform: Optional[torchvision.transforms.Compose] = None,
    device: torch.device = device,
):
    """Makes a prediction on an image and plots the result.

    Args:
        model (torch.nn.Module): Pre-trained PyTorch model.
        class_names (List[str]): List of class names for the classification task.
        image_path (str): Path to the image file to predict on.
        image_size (Tuple[int, int], optional): Resize dimensions for the image. Defaults to (224, 224).
        transform (Optional[torchvision.transforms.Compose], optional): Optional transform to apply to the image. Defaults to None.
        device (torch.device, optional): Device to run predictions on. Defaults to detected device (CUDA if available, else CPU).

    Returns:
        None. Displays the image with predicted class and probability.
    """

    # Open and display the image
    img = Image.open(image_path)

    # Use provided transform or create a default one (ImageNet normalization)
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    # Transform the image and add batch dimension
    transformed_img = transform(img).unsqueeze(dim=0).to(device)

    # Set model to evaluation mode and make a prediction
    model = model.to(device)
    model.eval()
    with torch.inference_mode():
        pred_logits = model(transformed_img)

    # Convert logits to probabilities using softmax
    pred_probs = torch.softmax(pred_logits, dim=1)

    # Get the predicted class index
    pred_label_idx = torch.argmax(pred_probs, dim=1).item()

    # Plot the image and prediction result
    plt.figure()
    plt.imshow(img)
    plt.title(f"Pred: {class_names[pred_label_idx]} | Prob: {pred_probs[0][pred_label_idx]:.3f}")
    plt.axis("off")
    plt.show()
