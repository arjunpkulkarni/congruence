import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_dataloaders(train_dir: str, test_dir: str, batch_size: int = 32, num_workers: int = 4):
    """
    Build DataLoaders for macro-expression classification using folder structure.
    Expected directory layout:
        train_dir/
            class_a/
            class_b/
            ...
        test_dir/
            class_a/
            class_b/
            ...
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_data = datasets.ImageFolder(train_dir, transform=transform)
    test_data = datasets.ImageFolder(test_dir, transform=transform)

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader, train_data.classes


