
from pathlib import Path
import json
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score
)
from PIL import Image

from preprocess import BASE_DIR, DATA_DIR, MEAN, STD


OUTPUT_DIR = BASE_DIR.parent / "output"
BATCH_SIZE = 64
NUM_EXPLANATIONS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        weights = torch.sigmoid(
            self.conv(torch.cat([avg, maximum], dim=1)) * 2
        )
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
        return self.spatial_attention(x)


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


def load_model():
    checkpoint = torch.load(
        OUTPUT_DIR / "best_model.pth",
        map_location=DEVICE,
        weights_only=True
    )

    classes = checkpoint["class_names"]
    image_size = checkpoint.get("image_size", 224)

    model = ButterflyCNN(len(classes)).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, classes, image_size


def load_test_data(classes, image_size):
    test_dir = DATA_DIR / "test"

    dataset = datasets.ImageFolder(
        test_dir,
        transform=transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])
    )

    if dataset.classes != classes:
        raise ValueError("Test class order differs from training.")

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    return dataset, loader


def predict(model, loader):
    y_true, y_pred, y_probs = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(DEVICE))
            probabilities = torch.softmax(logits, dim=1)

            y_true.extend(labels.numpy())
            y_pred.extend(probabilities.argmax(1).cpu().numpy())
            y_probs.extend(probabilities.cpu().numpy())

    return np.array(y_true), np.array(y_pred), np.array(y_probs)


def select_images(dataset):
    return np.random.default_rng().choice(
        len(dataset),
        size=min(NUM_EXPLANATIONS, len(dataset)),
        replace=False
    )


def prepare_image(image_path, image_size):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((image_size, image_size))

    tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])(image).unsqueeze(0).to(DEVICE)

    return image, tensor


def save_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(classes)))

    normalized = cm.astype(float) / np.maximum(
        cm.sum(axis=1, keepdims=True), 1
    )

    for matrix, filename, title, fmt in [
        (cm, "confusion_matrix.png", "Confusion Matrix", "d"),
        (
            normalized,
            "confusion_matrix_normalized.png",
            "Normalized Confusion Matrix",
            ".2f"
        )
    ]:
        size = max(12, min(24, len(classes) * 0.25))
        fig, ax = plt.subplots(figsize=(size, size))

        ConfusionMatrixDisplay(
            confusion_matrix=matrix,
            display_labels=classes
        ).plot(
            ax=ax,
            xticks_rotation=90,
            values_format=fmt,
            colorbar=False
        )

        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / filename, dpi=180)
        plt.close()


def save_predictions(dataset, indices, y_true, y_pred, probabilities, classes):
    fig, axes = plt.subplots(
        1, len(indices), figsize=(5 * len(indices), 5)
    )
    axes = np.atleast_1d(axes)

    for ax, idx in zip(axes, indices):
        image_path, _ = dataset.samples[idx]
        image = Image.open(image_path).convert("RGB")
        predicted = y_pred[idx]

        ax.imshow(image)
        ax.set_title(
            f"True: {classes[y_true[idx]]}\n"
            f"Predicted: {classes[predicted]}\n"
            f"Confidence: {probabilities[idx, predicted]:.1%}",
            color="green" if predicted == y_true[idx] else "red",
            fontsize=9
        )
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "prediction_examples.png", dpi=160)
    plt.close()


class GradCAM:
    def __init__(self, model):
        self.model = model
        self.activations = None

        self.handle = model.features[4].block[5].register_forward_hook(
            self.save_activation
        )

    def save_activation(self, module, inputs, output):
        self.activations = output

    def generate(self, image, class_idx):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image)

        gradients = torch.autograd.grad(
            logits[0, class_idx],
            self.activations
        )[0]

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)

        cam = torch.nn.functional.interpolate(
            cam,
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False
        )[0, 0]

        cam = cam.detach().cpu().numpy()
        cam -= cam.min()
        cam /= cam.max() + 1e-8

        return cam

    def close(self):
        self.handle.remove()


def save_gradcam(model, dataset, indices, classes, image_size):
    gradcam = GradCAM(model)

    fig, axes = plt.subplots(
        len(indices), 3, figsize=(12, 4 * len(indices))
    )
    axes = np.atleast_2d(axes)

    for row, idx in enumerate(indices):
        image_path, true_label = dataset.samples[idx]
        original, tensor = prepare_image(image_path, image_size)

        with torch.enable_grad():
            logits = model(tensor)
            predicted = int(logits.argmax(1).item())
            cam = gradcam.generate(tensor, predicted)

        original_array = np.asarray(original)

        axes[row, 0].imshow(original_array)
        axes[row, 0].set_title(f"Original: {classes[true_label]}")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(cam, cmap="jet", vmin=0, vmax=1)
        axes[row, 1].set_title("Grad-CAM heatmap")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(original_array)
        axes[row, 2].imshow(cam, cmap="jet", alpha=0.4, vmin=0, vmax=1)
        axes[row, 2].set_title(f"Predicted: {classes[predicted]}")
        axes[row, 2].axis("off")

    gradcam.close()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "gradcam_examples.png", dpi=200)
    plt.close()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Device:", DEVICE)

    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    model, classes, image_size = load_model()
    dataset, loader = load_test_data(classes, image_size)

    y_true, y_pred, probabilities = predict(model, loader)

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(
        y_true, y_pred, average="weighted", zero_division=0
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(classes)),
        target_names=classes,
        zero_division=0
    )

    print("\nTEST RESULTS")
    print("Test images:", len(dataset))
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print("\nClassification Report\n", report)

    (OUTPUT_DIR / "classification_report.txt").write_text(
        report, encoding="utf-8"
    )

    with open(OUTPUT_DIR / "test_results.json", "w", encoding="utf-8") as file:
        json.dump({
            "test_images": len(dataset),
            "num_classes": len(classes),
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1
        }, file, indent=2)

    indices = select_images(dataset)

    save_confusion_matrix(y_true, y_pred, classes)
    save_predictions(
        dataset, indices, y_true, y_pred, probabilities, classes
    )
    save_gradcam(model, dataset, indices, classes, image_size)

    print("\nEvaluation complete.")
    print("Outputs saved to:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
