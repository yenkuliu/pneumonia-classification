import argparse
import csv
import json
import os
import platform
import random
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from tqdm import tqdm
from protocol import filename_group, group_split, state_hash
from recipes import image_transforms, network, parameter_groups, stage


CLASSES = ["NORMAL", "PNEUMONIA"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


class ChestXrayDataset(Dataset):
    """Custom dataset for the Kaggle chest X-ray folder structure."""

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


def set_seed(seed, deterministic=False):
    if deterministic:
        # Set this before seeding or querying CUDA can initialize its context.
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_environment(output_dir, device):
    import PIL
    import sklearn
    import torchvision

    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "pillow": PIL.__version__,
        "device": str(device),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
    }
    if device.type == "cuda":
        environment["gpu"] = torch.cuda.get_device_name(device)
    with (output_dir / "environment.json").open("w") as stream:
        json.dump(environment, stream, indent=2)
    freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    (output_dir / "requirements-frozen.txt").write_text(freeze)
    source = Path(__file__).resolve()
    destination = output_dir / "train_snapshot.py"
    if source != destination.resolve():
        shutil.copy2(source, destination)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def labels_from_dataset(dataset):
    if isinstance(dataset, Subset):
        return [dataset.dataset.samples[i][1] for i in dataset.indices]
    return [label for _, label in dataset.samples]


def class_counts(dataset):
    labels = labels_from_dataset(dataset)
    counts = np.bincount(labels, minlength=len(CLASSES)).astype(int).tolist()
    return dict(zip(CLASSES, counts))


def class_weights_from_dataset(dataset, config):
    labels = labels_from_dataset(dataset)
    counts = np.bincount(labels, minlength=len(CLASSES)).astype(np.float32)
    strategy = config["class_weight_strategy"]
    if strategy == "manual":
        weights = np.array([config["manual_class_weights"][name] for name in CLASSES], dtype=np.float32)
    elif strategy == "inverse_frequency":
        weights = counts.sum() / (len(CLASSES) * counts)
    elif strategy == "sqrt_inverse_frequency":
        weights = np.sqrt(counts.sum() / (len(CLASSES) * counts))
    elif strategy == "none":
        weights = np.ones(len(CLASSES), dtype=np.float32)
    else:
        raise ValueError(f"Unknown class_weight_strategy: {strategy}")
    return torch.tensor(weights, dtype=torch.float32), counts.astype(int).tolist()


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


def run_epoch(model, loader, criterion, device, optimizer=None, freeze_batchnorm=False):
    is_train = optimizer is not None
    model.train(is_train)
    if is_train and freeze_batchnorm:
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
    total_loss = 0.0
    labels_all = []
    preds_all = []
    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        for images, labels in tqdm(loader, leave=False):
            images = images.to(device)
            labels = labels.to(device)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if is_train:
                loss.backward()
                optimizer.step()
            preds = torch.argmax(logits, dim=1)
            total_loss += loss.item() * images.size(0)
            labels_all.extend(labels.detach().cpu().numpy().tolist())
            preds_all.extend(preds.detach().cpu().numpy().tolist())

    metrics = compute_metrics(labels_all, preds_all)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


def collect_probs(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    labels_all = []
    probs_all = []

    with torch.no_grad():
        for images, labels in tqdm(loader, leave=False):
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            probs = torch.softmax(logits, dim=1)
            total_loss += loss.item() * images.size(0)
            labels_all.extend(labels.cpu().numpy().tolist())
            probs_all.extend(probs.cpu().numpy().tolist())

    return np.array(labels_all), np.array(probs_all), total_loss / len(loader.dataset)


def preds_at_threshold(probs, threshold):
    return (probs[:, 1] >= threshold).astype(int)


def save_history(history, output_dir):
    fieldnames = ["epoch"]
    for prefix in ["train", "val"]:
        for metric in ["loss", "accuracy", "balanced_accuracy", "f1", "precision", "recall", "specificity"]:
            fieldnames.append(f"{prefix}_{metric}")
    with (output_dir / "history.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def plot_curves(history, output_dir):
    epochs = [row["epoch"] for row in history]
    for metric, filename, title in [
        ("accuracy", "accuracy_curves.png", "Accuracy Trend"),
        ("balanced_accuracy", "balanced_accuracy_curves.png", "Balanced Accuracy Trend"),
        ("f1", "f1_curves.png", "F1-score Trend"),
        ("specificity", "specificity_curves.png", "Specificity Trend"),
    ]:
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, [row[f"train_{metric}"] for row in history], marker="o", label="Train")
        plt.plot(epochs, [row[f"val_{metric}"] for row in history], marker="o", label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel(metric.replace("_", " ").title())
        plt.title(title)
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=200)
        plt.close()


def save_confusion_matrix(labels, preds, output_dir, filename, title):
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES, cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=200)
    plt.close()
    return cm.tolist()


def write_predictions(dataset, labels, probs, preds, output_path, threshold):
    samples = dataset.dataset.samples if isinstance(dataset, Subset) else dataset.samples
    if isinstance(dataset, Subset):
        samples = [dataset.dataset.samples[i] for i in dataset.indices]

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image_path", "true_label", "pred_label", "prob_NORMAL", "prob_PNEUMONIA", "threshold", "is_correct"],
        )
        writer.writeheader()
        for (path, _), label, pred, prob in zip(samples, labels.tolist(), preds.tolist(), probs.tolist()):
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


def make_sampler(dataset, config):
    mode = config.get("sampler_strategy", "none")
    if mode == "none":
        return None

    labels = np.array(labels_from_dataset(dataset), dtype=np.int64)
    counts = np.bincount(labels, minlength=len(CLASSES)).astype(np.float32)
    class_weights = counts.sum() / (len(CLASSES) * counts)
    if mode == "weighted_balanced":
        sample_weights = class_weights[labels]
    elif mode == "weighted_normal_boost":
        class_weights[0] *= float(config.get("normal_boost", 1.25))
        sample_weights = class_weights[labels]
    else:
        raise ValueError(f"Unknown sampler_strategy: {mode}")

    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--deterministic", action="store_true",
                        help="Require deterministic operations for same-environment reruns")
    parser.add_argument("--validation-only", action="store_true",
                        help="Select the checkpoint on validation; evaluate the test split separately")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = json.load(open(args.config))
    config["deterministic"] = args.deterministic or config.get("deterministic", False)
    config["validation_only"] = args.validation_only or config.get("validation_only", False)
    if not config["validation_only"]:
        raise ValueError("Training requires validation-only checkpoint selection. Use evaluate.py for final test evaluation.")
    json.dump(config, open(output_dir / "config_snapshot.json", "w"), indent=2)

    set_seed(config["seed"], deterministic=config["deterministic"])
    device = get_device()
    if config.get("require_cuda", False) and device.type != "cuda":
        raise RuntimeError("This experiment requires CUDA; CPU/MPS fallback is disabled.")
    print(f"Using device: {device}")
    print(f"Deterministic algorithms: {torch.are_deterministic_algorithms_enabled()}")
    save_environment(output_dir, device)

    train_transform, eval_transform = image_transforms(config)
    full_train_aug = ChestXrayDataset(config["dataset_dir"], "train", train_transform)
    full_train_eval = ChestXrayDataset(config["dataset_dir"], "train", eval_transform)
    labels = labels_from_dataset(full_train_aug)
    indices = np.arange(len(labels))
    if config.get("split_policy") == "filename_groups_v1":
        train_indices, val_indices = group_split(full_train_aug.samples, config.get("split_seed", 820))
    else:
        train_indices, val_indices = train_test_split(
            indices, test_size=config["internal_validation_fraction"],
            stratify=labels, random_state=config.get("split_seed", config["seed"]))
        train_indices, val_indices = train_indices.tolist(), val_indices.tolist()

    train_dataset = Subset(full_train_aug, sorted(train_indices))
    val_dataset = Subset(full_train_eval, sorted(val_indices))

    split_record = {
        "train_counts": class_counts(train_dataset),
        "internal_val_counts": class_counts(val_dataset),
        "train_indices": train_dataset.indices,
        "internal_val_indices": val_dataset.indices,
    }
    if config.get("split_policy") == "filename_groups_v1":
        split_record["group_rule"] = "filename-derived proxy, not independently verified patient IDs"
        for key, subset in [("train_groups", train_dataset), ("validation_groups", val_dataset)]:
            split_record[key] = sorted({filename_group(*full_train_aug.samples[i]) for i in subset.indices})
    json.dump(split_record, open(output_dir / "split_indices.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in split_record.items() if "counts" in k}, indent=2))

    loader_kwargs = {
        "batch_size": config["batch_size"],
        "num_workers": config["num_workers"],
        "pin_memory": device.type == "cuda",
    }
    sampler = make_sampler(train_dataset, config)
    train_loader = DataLoader(train_dataset, shuffle=sampler is None, sampler=sampler, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    class_weights, raw_counts = class_weights_from_dataset(train_dataset, config)
    print(f"Class counts for loss weights: {dict(zip(CLASSES, raw_counts))}")
    print(f"Loss class weights: {class_weights.tolist()}")

    model = network(config["model"], config["num_classes"], config["dropout"]).to(device)
    state_audit = {"initial_state": state_hash(model.state_dict()), "epochs": []}
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        label_smoothing=config["label_smoothing"],
    )
    optimizer = optim.AdamW(parameter_groups(model, config), weight_decay=config["weight_decay"],
                           foreach=False, fused=False)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])

    best_value = -1.0
    best_epoch = 0
    best_state = None
    best_val_metrics = None
    stale_epochs = 0
    history = []
    selection_metric = config["selection_metric"]

    for epoch in range(1, config["epochs"] + 1):
        stage(model, config["model"], epoch, config["warmup_epochs"], config["tuning_scope"])
        if epoch == config["warmup_epochs"] + 1:
            stale_epochs = 0
        print(f"\nEpoch {epoch}/{config['epochs']}")
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer,
                                  freeze_batchnorm=(config.get("freeze_batchnorm", False) or
                                      epoch <= config["warmup_epochs"] or config["tuning_scope"] == "head"))
        val_metrics = run_epoch(model, val_loader, criterion, device)
        scheduler.step()

        row = {"epoch": epoch}
        row.update({f"train_{k}": v for k, v in train_metrics.items()})
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        history.append(row)
        state_audit["epochs"].append({"epoch": epoch, "state": state_hash(model.state_dict())})
        (output_dir / "state_audit.json").write_text(json.dumps(state_audit, indent=2))
        save_history(history, output_dir)
        plot_curves(history, output_dir)

        print(
            "train acc={train_accuracy:.4f} f1={train_f1:.4f} | "
            "val acc={val_accuracy:.4f} bal={val_balanced_accuracy:.4f} f1={val_f1:.4f}".format(**row)
        )

        if val_metrics[selection_metric] > best_value:
            best_value = val_metrics[selection_metric]
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            best_val_metrics = deepcopy(val_metrics)
            torch.save(best_state, output_dir / "best_model.pt")
            stale_epochs = 0
        else:
            stale_epochs += 1

        torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict()}, output_dir / "last_training_state.pt")

        if stale_epochs >= config["early_stopping_patience"]:
            print(f"Early stopping after {stale_epochs} stale epochs.")
            break

    model.load_state_dict(best_state)
    val_labels, val_probs, val_loss = collect_probs(model, val_loader, criterion, device)
    val_preds = preds_at_threshold(val_probs, 0.5)
    val_metrics = compute_metrics(val_labels, val_preds)
    val_metrics["loss"] = val_loss
    val_cm = save_confusion_matrix(
        val_labels,
        val_preds,
        output_dir,
        "validation_confusion_matrix.png",
        "Validation Confusion Matrix",
    )

    validation_metrics = {
        "experiment_name": config["experiment_name"],
        "evaluation_scope": "validation_only",
        "best_epoch": best_epoch,
        "epochs_trained": len(history),
        "early_stopping_triggered": stale_epochs >= config["early_stopping_patience"],
        "selection_metric": selection_metric,
        "best_selection_value": best_value,
        "best_validation_metrics_during_training": best_val_metrics,
        "threshold": 0.5,
        "validation_metrics_at_best_checkpoint": val_metrics,
        "validation_confusion_matrix": {"labels": CLASSES, "matrix": val_cm},
        "class_counts": {"train": class_counts(train_dataset), "val": class_counts(val_dataset)},
    }
    np.savez(output_dir / "validation_probs.npz", labels=val_labels, probabilities=val_probs)
    write_predictions(val_dataset, val_labels, val_probs, val_preds,
                      output_dir / "validation_predictions.csv", 0.5)
    json.dump(validation_metrics, open(output_dir / "metrics.json", "w"), indent=2)
    print(json.dumps(validation_metrics, indent=2))


if __name__ == "__main__":
    main()
