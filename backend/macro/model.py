import torch
import torch.nn as nn
from torchvision import models


def _get_resnet18_backbone():
    """
    Build a ResNet18 with ImageNet weights.
    Uses new weights API if available, falls back to pretrained=True for older torchvision.
    """
    try:
        weights = models.ResNet18_Weights.DEFAULT  # torchvision >= 0.13
        model = models.resnet18(weights=weights)
    except Exception:
        model = models.resnet18(pretrained=True)
    return model


def build_model(num_classes: int) -> nn.Module:
    """
    Create a classification model for macro-expressions.
    """
    model = _get_resnet18_backbone()
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def load_trained_backbone(weights_path: str, device: str = "cpu"):
    """
    Load the trained classifier weights and return a backbone that outputs embeddings.
    Returns (embedding_model, embedding_dim).
    """
    # Build the classifier to load matching state dict
    classifier = _get_resnet18_backbone()
    in_features = classifier.fc.in_features
    # Placeholder head (will be replaced by loaded state dict from training)
    classifier.fc = nn.Linear(in_features, 1)
    state = torch.load(weights_path, map_location=device)
    classifier.load_state_dict(state, strict=False)
    classifier.to(device)
    classifier.eval()

    # Convert to embedding backbone
    classifier.fc = nn.Identity()
    embedding_dim = in_features
    return classifier, embedding_dim


