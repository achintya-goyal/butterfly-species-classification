
from pathlib import Path
import copy
import json
import time
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import matplotlib.pyplot as plt

from preprocess import get_dataloaders, IMAGE_SIZE, BASE_DIR


OUTPUT_DIR = BASE_DIR.parent / "output"
EPOCHS = 50
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 0.0001
PATIENCE = 5


LABEL_SMOOTHING = 0.05


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()

        hidden = max(channels // reduction, 1)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        weights = self.fc(self.pool(x)).unsqueeze(-1).unsqueeze(-1)
        return x * weights


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        maximum = x.amax(dim=1, keepdim=True)
        combined = torch.cat([avg, maximum], dim=1)
        weights = torch.sigmoid(self.conv(combined) * 2)
        return x * weights


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, attention=False):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.channel_attention = (
            ChannelAttention(out_channels) if attention else nn.Identity()
        )
        self.spatial_attention = (
            SpatialAttention() if attention else nn.Identity()
        )

    def forward(self, x):
        x = self.block(x)
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class ButterflyCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256, attention=True),
            ConvBlock(256, 512, attention=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda"
            ):
                outputs = model(images)
                loss = criterion(outputs, labels)

            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        total_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def main():
    torch.manual_seed(42)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    train_loader, valid_loader, _, class_names = get_dataloaders()

    model = ButterflyCNN(len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=0.00001
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda"
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    best_accuracy = -1.0
    best_state = copy.deepcopy(model.state_dict())
    patience_counter = 0

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "valid_loss": [],
        "valid_accuracy": [],
        "learning_rate": []
    }

    start_time = time.time()

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion, device, optimizer, scaler
        )

        valid_loss, valid_accuracy = run_epoch(
            model, valid_loader, criterion, device
        )

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["valid_loss"].append(valid_loss)
        history["valid_accuracy"].append(valid_accuracy)
        history["learning_rate"].append(current_lr)

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Accuracy: {train_accuracy:.4f}"
        )
        print(
            f"Valid Loss: {valid_loss:.4f} | "
            f"Valid Accuracy: {valid_accuracy:.4f}"
        )
        print(f"Learning Rate: {current_lr:.7f}")

        if valid_accuracy > best_accuracy:
            best_accuracy = valid_accuracy
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0

            torch.save({
                "model_state_dict": best_state,
                "class_names": class_names,
                "image_size": IMAGE_SIZE,
                "architecture": "ButterflyCNN_Attention"
            }, OUTPUT_DIR / "best_model.pth")

            print("Best model saved.")
        else:
            patience_counter += 1

        with open(OUTPUT_DIR / "history.json", "w") as file:
            json.dump(history, file, indent=2)

        if patience_counter >= PATIENCE:
            print("Early stopping.")
            break

    model.load_state_dict(best_state)

    with open(OUTPUT_DIR / "class_names.json", "w") as file:
        json.dump(class_names, file, indent=2)

    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train")
    plt.plot(epochs, history["valid_loss"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_accuracy"], label="Train")
    plt.plot(epochs, history["valid_accuracy"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_curves.png", dpi=160)
    plt.close()

    print(f"\nBest validation accuracy: {best_accuracy:.4f}")
    print(f"Training time: {(time.time() - start_time) / 60:.2f} minutes")
    print("Saved model:", OUTPUT_DIR / "best_model.pth")


if __name__ == "__main__":
    main()
