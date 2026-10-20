import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from recipes import image_transforms, network


CLASSES = ["NORMAL", "PNEUMONIA"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


class ChestXrayDataset(Dataset):
    """Custom dataset for train/val/test chest X-ray folders."""

    def __init__(self, root, split, transform=None):
        self.root = Path(root).expanduser().resolve()
        self.split = split
        self.transform = transform
        self.samples = []

        split_dir = self.root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Missing split directory: {split_dir}")

        for label, class_name in enumerate(CLASSES):
            class_dir = split_dir / class_name
            if not class_dir.exists():
                raise FileNotFoundError(f"Missing class directory: {class_dir}")
            for image_path in sorted(class_dir.rglob("*")):
                if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append((image_path, label))

        if not self.samples:
            raise RuntimeError(f"No images found under {split_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_metrics(labels, preds):
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    recall = recall_score(labels, preds, pos_label=1, zero_division=0)
    return {
        "accuracy": accuracy_score(labels, preds),
        "balanced_accuracy": (recall + specificity) / 2.0,
        "f1": f1_score(labels, preds, pos_label=1, zero_division=0),
        "precision": precision_score(labels, preds, pos_label=1, zero_division=0),
        "recall": recall,
        "specificity": specificity,
    }


def save_confusion_matrix(labels, preds, output_dir):
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES, cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=200)
    plt.close()
    return cm.tolist()


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    model.load_state_dict(checkpoint)
    return model


def predict_single_image(model, transform, image_path, threshold, device):
    model.eval()
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred = int(probs[1] >= threshold)
    return {
        "image_path": str(image_path),
        "pred_label": CLASSES[pred],
        "prob_NORMAL": float(probs[0]),
        "prob_PNEUMONIA": float(probs[1]),
        "threshold": threshold,
    }


def evaluate_split(model, transform, data_dir, split, threshold, batch_size, num_workers, output_dir, device):
    dataset = ChestXrayDataset(data_dir, split, transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    labels_all = []
    probs_all = []
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(loader, leave=False):
            images = images.to(device)
            probs = torch.softmax(model(images), dim=1).cpu().numpy()
            labels_all.extend(labels.numpy().tolist())
            probs_all.extend(probs.tolist())

    labels = np.array(labels_all)
    probs = np.array(probs_all)
    preds = (probs[:, 1] >= threshold).astype(int)
    metrics = compute_metrics(labels, preds)
    cm = save_confusion_matrix(labels, preds, output_dir)

    with (output_dir / "predictions.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image_path", "true_label", "pred_label", "prob_NORMAL", "prob_PNEUMONIA", "threshold", "is_correct"],
        )
        writer.writeheader()
        for (path, _), label, pred, prob in zip(dataset.samples, labels.tolist(), preds.tolist(), probs.tolist()):
            writer.writerow(
                {
                    "image_path": str(path),
                    "true_label": CLASSES[label],
                    "pred_label": CLASSES[pred],
                    "prob_NORMAL": prob[0],
                    "prob_PNEUMONIA": prob[1],
                    "threshold": threshold,
                    "is_correct": int(label == pred),
                }
            )

    result = {
        "split": split,
        "threshold": threshold,
        "metrics": metrics,
        "confusion_matrix": {"labels": CLASSES, "matrix": cm},
    }
    json.dump(result, open(output_dir / "metrics.json", "w"), indent=2)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--image", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output-dir", default="outputs/inference")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = json.load(open(args.config))
    threshold = args.threshold

    device = get_device()
    if config.get("require_cuda", False) and device.type != "cuda":
        raise RuntimeError("CUDA required; no CPU/MPS fallback.")
    _, transform = image_transforms(config)
    model = network(config["model"], config["num_classes"], config["dropout"], pretrained=False).to(device)
    model = load_checkpoint(model, args.checkpoint, device)

    if args.image:
        result = predict_single_image(model, transform, Path(args.image), threshold, device)
        print(json.dumps(result, indent=2))
        json.dump(result, open(output_dir / "single_image_prediction.json", "w"), indent=2)
        return

    data_dir = args.data_dir or config["dataset_dir"]
    result = evaluate_split(
        model,
        transform,
        data_dir,
        args.split,
        threshold,
        args.batch_size,
        args.num_workers,
        output_dir,
        device,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
