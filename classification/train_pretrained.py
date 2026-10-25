
from pathlib import Path
import copy
import json
import time
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, classification_report

from preprocess import get_dataloaders, BASE_DIR


OUTPUT_DIR = BASE_DIR.parent / "output"
EPOCHS = 20
HEAD_EPOCHS = 3
BATCH_SIZE = 64
HEAD_LR = 0.001
BACKBONE_LR = 0.0001
WEIGHT_DECAY = 0.0001
PATIENCE = 7
LABEL_SMOOTHING = 0.05


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

    train_loader, valid_loader, test_loader, class_names = get_dataloaders()

    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, len(class_names))
    )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss(
        label_smoothing=LABEL_SMOOTHING
    )

    optimizer = AdamW(
        model.fc.parameters(),
        lr=HEAD_LR,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = None

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
        if epoch == HEAD_EPOCHS:
            for param in model.layer4.parameters():
                param.requires_grad = True

            optimizer = AdamW(
                [
                    {"params": model.layer4.parameters(), "lr": BACKBONE_LR},
                    {"params": model.fc.parameters(), "lr": HEAD_LR * 0.2}
                ],
                weight_decay=WEIGHT_DECAY
            )

            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=max(EPOCHS - HEAD_EPOCHS, 1),
                eta_min=0.000005
            )

            print("Unfroze ResNet18 layer4 for fine-tuning.")

        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion, device, optimizer, scaler
        )

        valid_loss, valid_accuracy = run_epoch(
            model, valid_loader, criterion, device
        )

        if scheduler is not None:
            scheduler.step()

        current_lr = optimizer.param_groups[-1]["lr"]

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
                "architecture": "ResNet18",
                "weights": "ImageNet",
                "image_size": 224
            }, OUTPUT_DIR / "best_resnet18.pth")

            print("Best ResNet18 model saved.")
        else:
            patience_counter += 1

        with open(OUTPUT_DIR / "resnet18_history.json", "w") as file:
            json.dump(history, file, indent=2)

        if patience_counter >= PATIENCE:
            print("Early stopping.")
            break

    model.load_state_dict(best_state)
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device, non_blocking=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda"
            ):
                outputs = model(images)

            y_pred.extend(outputs.argmax(1).cpu().tolist())
            y_true.extend(labels.tolist())

    test_accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_f1 = f1_score(
        y_true, y_pred, average="weighted", zero_division=0
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0
    )

    print("\nRESNET18 TEST RESULTS")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(report)

    (OUTPUT_DIR / "resnet18_classification_report.txt").write_text(
        report, encoding="utf-8"
    )

    with open(OUTPUT_DIR / "resnet18_test_results.json", "w") as file:
        json.dump({
            "architecture": "ResNet18 pretrained on ImageNet",
            "test_images": len(y_true),
            "test_accuracy": test_accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "best_validation_accuracy": best_accuracy
        }, file, indent=2)

    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train")
    plt.plot(epochs, history["valid_loss"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("ResNet18 Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_accuracy"], label="Train")
    plt.plot(epochs, history["valid_accuracy"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("ResNet18 Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "resnet18_training_curves.png", dpi=160)
    plt.close()

    print(f"\nTraining time: {(time.time() - start_time) / 60:.2f} minutes")
    print("Checkpoint:", OUTPUT_DIR / "best_resnet18.pth")


if __name__ == "__main__":
    main()
