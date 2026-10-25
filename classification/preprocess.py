
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
IMAGE_SIZE = 224
BATCH_SIZE = 64
NUM_WORKERS = 0
SEED = 42
MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(12),
        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.15,
            hue=0.03
        ),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

    return train_transform, eval_transform


def get_dataloaders():
    set_seed()

    train_dir = DATA_DIR / "train"
    valid_dir = DATA_DIR / "valid"
    test_dir = DATA_DIR / "test"

    if not train_dir.is_dir():
        raise FileNotFoundError(f"Training folder not found: {train_dir}")

    train_transform, eval_transform = get_transforms()

    train_dataset = datasets.ImageFolder(
        train_dir,
        transform=train_transform
    )

    class_names = train_dataset.classes
    expected_classes = set(class_names)

    if len(class_names) < 2:
        raise ValueError("At least two classes are required.")

    print("Classes:", len(class_names))
    print("Training images:", len(train_dataset))

    for name, folder in [("valid", valid_dir), ("test", test_dir)]:
        if not folder.is_dir():
            raise FileNotFoundError(f"{name} folder not found: {folder}")

        dataset = datasets.ImageFolder(folder)

        if set(dataset.classes) != expected_classes:
            raise ValueError(f"Class folders differ in {name}.")

        print(f"{name.capitalize()} images:", len(dataset))

    valid_dataset = datasets.ImageFolder(
        valid_dir,
        transform=eval_transform
    )

    test_dataset = datasets.ImageFolder(
        test_dir,
        transform=eval_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    print("Image size:", IMAGE_SIZE)
    print("Batch size:", BATCH_SIZE)
    print("CUDA available:", torch.cuda.is_available())

    return train_loader, valid_loader, test_loader, class_names


if __name__ == "__main__":
    train_loader, valid_loader, test_loader, class_names = get_dataloaders()

    images, labels = next(iter(train_loader))

    print("Image batch:", images.shape)
    print("Label batch:", labels.shape)
    print("Classes:", class_names)
    print("Device:", "cuda" if torch.cuda.is_available() else "cpu")
