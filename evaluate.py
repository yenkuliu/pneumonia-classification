"""Check checkpoint and validation probabilities before fixed-0.5 test inference."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import train
from protocol import state_hash
from recipes import image_transforms, network


def assessment_device():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU/MPS fallback is disabled.")
    return torch.device("cuda")


def evaluate(attempt, data, output):
    config = json.loads((attempt / "config_snapshot.json").read_text())
    metrics = json.loads((attempt / "metrics.json").read_text())
    if metrics["evaluation_scope"] != "validation_only":
        raise RuntimeError("Expected training with validation-only checkpoint selection.")
    train.set_seed(config["seed"], deterministic=True)
    device = assessment_device()
    model = network(config["model"], config["num_classes"], config["dropout"], pretrained=False).to(device)
    state = torch.load(attempt / "best_model.pt", map_location="cpu", weights_only=True)
    audit = json.loads((attempt / "state_audit.json").read_text())
    recorded = next(row["state"] for row in audit["epochs"] if row["epoch"] == metrics["best_epoch"])
    if state_hash(state) != recorded:
        raise RuntimeError("Best checkpoint differs from its recorded epoch state.")
    model.load_state_dict(state)
    _, transform = image_transforms(config)
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([
        config["manual_class_weights"][name] for name in train.CLASSES], device=device),
        label_smoothing=config["label_smoothing"])
    split = json.loads((attempt / "split_indices.json").read_text())
    validation = Subset(train.ChestXrayDataset(data, "train", transform), split["internal_val_indices"])

    def loader(dataset):
        return DataLoader(dataset, batch_size=config["batch_size"], shuffle=False,
                          num_workers=config["num_workers"], pin_memory=device.type == "cuda")

    labels, probabilities, _ = train.collect_probs(model, loader(validation), criterion, device)
    with np.load(attempt / "validation_probs.npz") as saved:
        checks = dict(labels_exact=np.array_equal(labels, saved["labels"]),
                      probabilities_exact=np.array_equal(probabilities, saved["probabilities"]))
    output.mkdir(parents=True, exist_ok=False)
    (output / "validation_check.json").write_text(json.dumps(checks, indent=2) + "\n")
    if not all(checks.values()):
        raise RuntimeError("Validation inference differs from training. Test has NOT been evaluated.")

    dataset = train.ChestXrayDataset(data, "test", transform)
    labels, probabilities, _ = train.collect_probs(model, loader(dataset), criterion, device)
    predictions = train.preds_at_threshold(probabilities, 0.5)
    matrix = train.save_confusion_matrix(labels, predictions, output,
                                         "confusion_matrix.png", "Final test (threshold 0.5)")
    train.write_predictions(dataset, labels, probabilities, predictions, output / "predictions.csv", 0.5)
    np.savez(output / "probabilities.npz", labels=labels, probabilities=probabilities)
    result = dict(model=config["model"], split="test", threshold=0.5,
                  metrics=train.compute_metrics(labels, predictions),
                  confusion_matrix=dict(labels=train.CLASSES, matrix=matrix),
                  validation_probabilities_exact=True, best_state_hash=recorded)
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.attempt.resolve(), args.data_dir.expanduser().resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
