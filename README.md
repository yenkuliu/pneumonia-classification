# AIMI Lab 1: Pneumonia Classification

This repository contains the training and inference implementation used for the report. It trains **ResNet34 and ResNet50**, with **DenseNet121** as an additional model. A custom PyTorch Dataset reads NORMAL/PNEUMONIA images. The reproduction command starts each model from its torchvision ImageNet initialization, reinitializes the two-class head, and selects a checkpoint on internal validation balanced accuracy.

## Reported CUDA Results

| Model | Test accuracy | Test F1 | FP / FN | Best epoch | Early stopping |
|---|---:|---:|---:|---:|---:|
| DenseNet121 | 0.9054487179 | 0.9291716687 | 56 / 3 | 9 | 14 |
| ResNet34 | 0.8894230769 | 0.9181494662 | 66 / 3 | 9 | 14 |
| ResNet50 | 0.8605769231 | 0.8989547038 | 84 / 3 | 9 | 14 |

Confusion matrix rows are true labels and columns are predictions, both ordered `[NORMAL, PNEUMONIA]`. The DenseNet121 matrix is `[[178, 56], [3, 387]]`, giving 565 correct predictions out of 624 images. `reference/report_metrics.json` stores all six reported metrics and the three matrices.

## 1. Environment

Reference: Linux x86_64, NVIDIA Tesla T4, Python **3.12.14**, PyTorch **2.14.0+cu126**, torchvision **0.29.0+cu126**, NumPy **2.5.3**, SciPy **1.18.1**, scikit-learn **1.9.1**, Pillow **12.3.0**. Other installed package versions are pinned in `requirements-lock.txt`, copied from the resolved Linux environment constraints.

After downloading/cloning this repository, run these commands **on the NVIDIA GPU computer**, from the repository root:

```bash
bash setup_cuda.sh
source .venv/bin/activate
```

The setup script creates a repository-local environment and installs the official CUDA 12.6 wheels. It does not delete or modify an existing parent environment. It uses an existing `uv`, or bootstraps `uv` with `python3 -m venv`. A working internet connection is needed for packages and the first ImageNet weight download. If `venv` is unavailable on Ubuntu, ask the administrator to install `python3-venv`. If CUDA initialization fails, inspect `nvidia-smi` and ask the administrator about driver compatibility; do not silently switch to CPU.

Use `bash run.sh ...` to set `PYTHONHASHSEED=820`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, and OMP/MKL/OpenBLAS thread counts to 4 before Python starts. CUDA training uses deterministic FP32 operations, disabled TF32, and AdamW with `foreach=False, fused=False`.

Exact reproduction requires the reference hardware and software environment. The default preflight rejects mismatched core package/Python/CUDA versions and requires CUDA for training. It records GPU/driver information and warns if the GPU is not a Tesla T4. `--allow-environment-differences` permits version differences but still requires CUDA; identical results are not guaranteed in a different environment.

## 2. Dataset

Dataset: [Kaggle Chest X-Ray Pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).

Set `DATA_DIR` to the directory that directly contains `train` and `test`, not to its parent:

```text
chest_xray/
  train/NORMAL/       1341 images
  train/PNEUMONIA/    3875 images
  test/NORMAL/         234 images
  test/PNEUMONIA/      390 images
  val/                not used
```

For an existing dataset:

```bash
export DATA_DIR=/absolute/path/to/chest_xray
```

Alternatively, download using a separate helper environment so the locked training environment stays unchanged:

```bash
python3 -m venv .download-venv
.download-venv/bin/python -m pip install kagglehub
.download-venv/bin/python download_dataset.py
export DATA_DIR="$(cat dataset_path.txt)"
```

The helper downloads the currently available dataset; the preflight checks every image against the reference content checksum and rejects changed versions, missing images, duplicates, or different files. Reference data was Kaggle version 2. Downloaded data is not included in this repository.

Only the original `train` folder is split, using the first fold of `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=820)`. Filename-derived group identifiers do not overlap across training and validation; these are proxies, not independently verified patient identifiers.

| Partition | NORMAL | PNEUMONIA | Total |
|---|---:|---:|---:|
| Training | 1073 | 3100 | 4173 |
| Internal validation | 268 | 775 | 1043 |
| Official test | 234 | 390 | 624 |

The preflight checks image counts, relative filenames, file sizes, image-content SHA256, split indices, and partition counts. The full dataset checksum is `4d0013e524bee56f2b2062d0380d51943eb1d4c6b664374a062ceafde46d0e27`; the split checksum is `4b8289ab7efe71eee7fb740b4a47c3aaff9084116acd0204dfd98a20998322b0`.

Check prerequisites without training or test inference:

```bash
bash run.sh check --data-dir "$DATA_DIR"
```

## 3. Reproduce Training and Final Evaluation

Train all three models and repeat DenseNet121 independently:

```bash
bash run.sh reproduce \
  --data-dir "$DATA_DIR" \
  --output-dir outputs/final_cuda \
  --repeat-models densenet121
```

Each model is trained from ImageNet initialization. Training and independent evaluation run in separate Python subprocesses. The evaluator checks checkpoint state hashes and saved validation probabilities before evaluating the test split. Checkpoint selection uses validation balanced accuracy, and validation and final-test predictions use a fixed threshold of 0.5.

To train only the required ResNet comparison:

```bash
bash run.sh reproduce --data-dir "$DATA_DIR" \
  --output-dir outputs/resnets --models resnet34 resnet50
```

To train only the best model:

```bash
bash run.sh reproduce --data-dir "$DATA_DIR" \
  --output-dir outputs/densenet --models densenet121 \
  --repeat-models densenet121
```

Repeat any selected architectures using `--repeat-models resnet34 resnet50 densenet121`. This doubles the selected models' training work. Comparisons use tensor-state hashes and exact histories/probabilities, not serialized checkpoint byte identity alone.

For a disconnect-safe remote job:

```bash
nohup bash run.sh reproduce --data-dir "$DATA_DIR" \
  --output-dir outputs/final_cuda --repeat-models densenet121 \
  > reproduction.log 2>&1 < /dev/null &
echo "PID: $!"
tail -F reproduction.log
```

Do not launch this a second time if the same job is already running. `Ctrl+C` while following `tail` stops only the viewer. `Ctrl+C` during foreground reproduction interrupts its active subprocess; it is not a pause. To resume later, use the **same** model list/data/environment/output path:

```bash
bash run.sh reproduce --data-dir "$DATA_DIR" \
  --output-dir outputs/final_cuda --repeat-models densenet121 --resume
```

Resume verifies saved inputs and artifact hashes, skips fully completed model evaluations, and restarts an interrupted model from ImageNet in a new attempt directory. It does **not** resume partway through an optimizer trajectory. Interrupted attempts are retained. Output directories are locked against simultaneous jobs. A nonempty directory is refused unless `--resume` is supplied.

### Shared Training Configuration

`configs/shared.json` is copied from the final experiment configuration. The reproduction entry point adds only model name, experiment label, and the dataset path. All other fine-tuning settings are shared:

| Setting | Value |
|---|---|
| Input/preprocessing | Full-frame resize to 256 x 256; no crop/autocontrast |
| Normalization | ImageNet RGB mean/std |
| Training augmentation | Flip 0.5, rotation 10 degrees, brightness/contrast jitter 0.12 |
| Optimizer | AdamW, backbone LR 1e-5, head LR 5e-4, weight decay 5e-4 |
| Class weights | NORMAL 1.5 / PNEUMONIA 1.0; no weighted sampler |
| Loss / dropout | Cross entropy, smoothing 0; classifier dropout 0.25 |
| Warmup | Head only for 2 epochs, then all parameters fine-tuned |
| Batch normalization | Running statistics frozen |
| Batch / workers | 16 / 2 |
| Schedule | CosineAnnealingLR, maximum 20 epochs |
| Checkpoint / stopping | Validation balanced accuracy; patience 5 |
| Seed / final threshold | 820 / 0.5 |

## 4. Outputs and Inference

Each output run contains:

```text
status.json                       completion/reference/repetition checks
preflight.json                    versions, source hashes, data/split checks
dataset_manifest.json             content fingerprints for every input image
source/                           frozen source/configuration for this run
comparison.json                   newly measured model metrics and attempt paths
reference_comparison.json         comparison with the reported metrics
repetition_checks.json            present when repetitions are requested
summary.csv                       compact final comparison
figures/                          PDF training/validation curves, DenseNet121 confusion matrix, score comparison
models/<model>/attempt_.../
  train.log                       training output
  history.csv                     every completed epoch
  config_snapshot.json            full training configuration
  split_indices.json              exact train/internal-validation indices
  environment.json                training environment
  requirements-frozen.txt         actual installed packages
  state_audit.json                initial and per-epoch tensor hashes
  best_model.pt                   validation-selected checkpoint
  last_training_state.pt          final model/optimizer/scheduler state
  validation_probs.npz            saved validation probabilities
  validation_predictions.csv      validation predictions at threshold 0.5
  final_test/metrics.json          actual fixed-0.5 final metrics
  final_test/probabilities.npz     actual per-image probabilities
  final_test/predictions.csv       filenames, probabilities, labels, correctness
```

`LAST_RUN.txt` points to the most recent training run. `status.json` distinguishes `running`, `completed`, `completed_with_mismatch`, `failed`, and `interrupted`. A complete run exits **0** only if all requested report-score and repetition checks pass; a measured mismatch exits **2**, preserving the measured results.

Read an attempt's `evaluation.log` or `final_test/metrics.json` to inspect its final test results.

Independent test inference on a generated checkpoint:

```bash
bash run.sh inference \
  --checkpoint /absolute/path/to/attempt/best_model.pt \
  --config /absolute/path/to/attempt/config_snapshot.json \
  --data-dir "$DATA_DIR" --split test \
  --output-dir outputs/independent_inference
```

This command uses deterministic inference, default threshold 0.5, and batch size 16. `--image /path/to/image.jpeg` instead predicts one image. Use only trusted checkpoints. During reproduction, `evaluate.py` also verifies that validation probabilities match the saved training output before test evaluation.

Regenerate comparison PDFs from an existing run:

```bash
bash run.sh plots --run outputs/final_cuda
```

Training curves use the stored per-epoch training metrics, with augmentation/dropout active. Validation curves use evaluation mode. Final test metrics use the best validation checkpoint. Figures are generated from the run's measured histories and predictions. When DenseNet121 is included, its final-test confusion matrix is saved as `figures/densenet121_confusion_matrix.pdf` with an image-count colorbar. ResNet-only runs generate the four curves and score comparison without a confusion-matrix figure.
