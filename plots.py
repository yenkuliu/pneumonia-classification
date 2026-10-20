"""Render comparison figures from newly generated histories and test predictions."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

LABELS = {"resnet34": "ResNet34", "resnet50": "ResNet50", "densenet121": "DenseNet121"}
COLORS = {"resnet34": "#187a58", "resnet50": "#bd3047", "densenet121": "#3e65a2"}


def render(run):
    models = json.loads((run / "comparison.json").read_text())
    order = [name for name in ["resnet34", "resnet50", "densenet121"] if name in models]
    if not order:
        raise ValueError("No completed models to plot.")
    target = run / "figures"
    target.mkdir(exist_ok=True)
    for filename in ["confusion_matrices.pdf", "architecture_input_accuracy_heatmap.pdf"]:
        (target / filename).unlink(missing_ok=True)
    histories = {}
    for name in order:
        with (run / models[name]["attempt"] / "history.csv").open() as stream:
            histories[name] = list(csv.DictReader(stream))
    plt.rcParams.update({"pdf.fonttype": 42, "font.size": 11})
    for split, metric in [("train", "accuracy"), ("train", "f1"), ("val", "accuracy"), ("val", "f1")]:
        figure, axis = plt.subplots(figsize=(8, 4.6))
        for name in order:
            rows = histories[name]
            axis.plot([int(row["epoch"]) for row in rows],
                      [float(row[f"{split}_{metric}"]) for row in rows],
                      marker="o", markersize=3, color=COLORS[name], label=LABELS[name])
        axis.set(xlabel="Epoch", ylabel="Accuracy" if metric == "accuracy" else "F1-score", ylim=(0, 1.02))
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.grid(alpha=0.2)
        axis.legend()
        figure.tight_layout()
        figure.savefig(target / f"{split}_{metric}.pdf", bbox_inches="tight")
        plt.close(figure)
    if "densenet121" in models:
        matrix = np.asarray(models["densenet121"]["confusion_matrix"])
        figure, axis = plt.subplots(figsize=(5.8, 4.5))
        heatmap = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=390)
        colorbar = figure.colorbar(heatmap, ax=axis, fraction=0.046, pad=0.04,
                                  ticks=[0, 100, 200, 300, 390])
        colorbar.set_label("Image count")
        for i in range(2):
            for j in range(2):
                axis.text(j, i, str(matrix[i, j]), ha="center", va="center",
                          color="white" if matrix[i, j] > 195 else "black", fontsize=15)
        axis.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["NORMAL", "PNEUMONIA"],
                 yticklabels=["NORMAL", "PNEUMONIA"], xlabel="Predicted label", ylabel="True label",
                 title="DenseNet121")
        figure.tight_layout()
        figure.savefig(target / "densenet121_confusion_matrix.pdf", bbox_inches="tight")
        plt.close(figure)
    else:
        (target / "densenet121_confusion_matrix.pdf").unlink(missing_ok=True)
    figure, axis = plt.subplots(figsize=(8, 4.6))
    positions = np.arange(len(order))
    for offset, metric, label, color in [(-0.18, "accuracy", "Test accuracy", "#187a58"),
                                         (0.18, "f1", "Test F1-score", "#3e65a2")]:
        bars = axis.bar(positions + offset, [models[name]["metrics"][metric] for name in order],
                        width=0.34, label=label, color=color)
        axis.bar_label(bars, fmt="%.4f", padding=4, fontsize=10)
    axis.set(xticks=positions, xticklabels=[LABELS[name] for name in order], ylabel="Score", ylim=(0, 1.1))
    axis.legend(frameon=False, ncol=2, loc="upper left")
    axis.set_axisbelow(True)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(target / "test_accuracy_f1.pdf", bbox_inches="tight")
    plt.close(figure)
    with (run / "summary.csv").open("w", newline="") as stream:
        keys = ["model", "threshold", "best_epoch", "epochs_trained", "accuracy", "f1",
                "balanced_accuracy", "precision", "recall", "specificity", "false_positives", "false_negatives"]
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for name in order:
            row = models[name]
            writer.writerow(dict(model=LABELS[name], threshold=row["threshold"], best_epoch=row["best_epoch"],
                epochs_trained=row["epochs_trained"], **row["metrics"],
                false_positives=row["confusion_matrix"][0][1], false_negatives=row["confusion_matrix"][1][0]))
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    print(render(args.run.expanduser().resolve()))
