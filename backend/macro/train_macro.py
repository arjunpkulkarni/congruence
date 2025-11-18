import json
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .dataset import get_dataloaders
from .model import build_model


HERE = Path(__file__).resolve().parent
DATASET_ROOT = HERE / "MACROEXPRESSIONS"
TRAIN_DIR = str(DATASET_ROOT / "train")
TEST_DIR = str(DATASET_ROOT / "test")
MODEL_WEIGHTS_PATH = HERE / "macro_expression_model.pth"
CLASSES_JSON_PATH = HERE / "macro_expression_classes.json"


def _evaluate(model: torch.nn.Module, data_loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / max(total, 1)


def train(epochs: int = 12, batch_size: int = 32, lr: float = 1e-4, num_workers: int = 4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader, classes = get_dataloaders(
        TRAIN_DIR, TEST_DIR, batch_size=batch_size, num_workers=num_workers
    )
    model = build_model(len(classes))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / max(len(train_loader), 1)
        val_acc = _evaluate(model, test_loader, device)
        print(f"Epoch {epoch + 1}/{epochs} - loss: {avg_loss:.4f} - val_acc: {val_acc:.4f}")

    # Save weights and classes
    torch.save(model.state_dict(), str(MODEL_WEIGHTS_PATH))
    with open(CLASSES_JSON_PATH, "w") as f:
        json.dump(classes, f)

    print(f"Saved weights to {MODEL_WEIGHTS_PATH}")
    print(f"Saved classes to {CLASSES_JSON_PATH} -> {classes}")


if __name__ == "__main__":
    train()


