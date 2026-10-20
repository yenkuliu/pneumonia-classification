"""Train and evaluate the three reported pneumonia classification models."""

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys


SETTINGS = dict(MPLBACKEND="Agg", PYTHONHASHSEED="820", PYTHONUNBUFFERED="1",
                CUBLAS_WORKSPACE_CONFIG=":4096:8", OMP_NUM_THREADS="4",
                MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
if __name__ == "__main__" and os.environ.get("PYTHONHASHSEED") != "820":
    os.environ.update(SETTINGS)
    os.execv(sys.executable, [sys.executable, *sys.argv])
os.environ.update(SETTINGS)

ROOT = Path(__file__).resolve().parent
MODELS = ["densenet121", "resnet34", "resnet50"]


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hashes():
    expected = read(ROOT / "reference/source_hashes.json")
    actual = {name: file_hash(ROOT / name) for name in expected}
    if expected != actual:
        raise RuntimeError("Packaged source/configuration changed; restore the supplied files before reproduction.")
    return actual


def inspect_environment():
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    available = torch.cuda.is_available()
    driver = None
    if available:
        x = torch.ones(4, 4, device="cuda", requires_grad=True)
        (x @ x).sum().backward()
        torch.cuda.synchronize()
        try:
            driver = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,driver_version",
                "--format=csv,noheader"], text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            driver = "unavailable"
    return dict(python=platform.python_version(), platform=platform.platform(),
                packages={name: importlib.metadata.version(name) for name in
                          ["torch", "torchvision", "numpy", "scipy", "scikit-learn", "Pillow"]},
                cuda=torch.version.cuda, cuda_available=available,
                gpu=torch.cuda.get_device_name(0) if available else None,
                cudnn=torch.backends.cudnn.version(), driver=driver, settings=SETTINGS)


def environment_differences(actual, reference):
    differences = {}
    for name in ["python", "cuda"]:
        if actual[name] != reference[name]:
            differences[name] = dict(actual=actual[name], expected=reference[name])
    for name, version in reference["packages"].items():
        if actual["packages"][name] != version:
            differences[name] = dict(actual=actual["packages"][name], expected=version)
    if not actual["cuda_available"]:
        differences["device"] = dict(actual="CUDA unavailable", expected="CUDA")
    return differences


def inspect_dataset(data):
    from train import ChestXrayDataset, CLASSES
    from protocol import group_split, split_hash
    data = Path(data).expanduser().resolve()
    aggregate = hashlib.sha256()
    rows, counts = [], {}
    datasets = {}
    for split in ["train", "test"]:
        dataset = datasets[split] = ChestXrayDataset(data, split)
        counts[split] = {name: 0 for name in CLASSES}
        for path, label in dataset.samples:
            counts[split][CLASSES[label]] += 1
            row = [path.relative_to(data).as_posix(), path.stat().st_size, file_hash(path)]
            rows.append(row)
            aggregate.update((json.dumps(row) + "\n").encode())
    a, b = group_split(datasets["train"].samples, 820)
    split = dict(train_indices=a, internal_val_indices=b)
    partition_counts = {}
    for name, indices in [("train", a), ("validation", b)]:
        partition_counts[name] = {label: sum(datasets["train"].samples[i][1] == index
            for i in indices) for index, label in enumerate(CLASSES)}
    return dict(sha256=aggregate.hexdigest(), counts=counts,
                partition_counts=partition_counts, split_sha256=split_hash(split), files=rows)


def validate_dataset(actual, reference):
    for field, expected in [("sha256", reference["dataset_sha256"]),
                            ("split_sha256", reference["split_sha256"]),
                            ("counts", reference["counts"]),
                            ("partition_counts", reference["partition_counts"])]:
        if actual[field] != expected:
            raise RuntimeError(f"Dataset/split mismatch in {field}: {actual[field]} != {expected}")


def execute(command, log):
    save(log.with_suffix(".command.json"), command)
    with log.open("x") as stream:
        child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        seen = None
        try:
            while True:
                try:
                    code = child.wait(timeout=5)
                    break
                except subprocess.TimeoutExpired:
                    history = log.parent / "history.csv"
                    try:
                        with history.open() as source:
                            rows = list(csv.DictReader(source))
                        if rows and rows[-1].get("epoch") != seen:
                            row = rows[-1]
                            seen = row["epoch"]
                            print(f"  epoch {seen}: train acc={float(row['train_accuracy']):.4f}, "
                                  f"val acc={float(row['val_accuracy']):.4f}", flush=True)
                    except (OSError, KeyError, ValueError):
                        pass
        except BaseException:
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    child.wait()
            raise
    if code:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        raise RuntimeError(f"Command failed ({code}); inspect {log}")


def check_completed(row, output):
    for relative, expected in row["artifacts"].items():
        if file_hash(output / relative) != expected:
            raise RuntimeError(f"Saved artifact changed: {relative}")


def run_model(output, name, data, job):
    folder = output / "models" / job
    folder.mkdir(parents=True, exist_ok=True)
    completed = folder / "completed.json"
    if completed.exists():
        row = read(completed)
        check_completed(row, output)
        print(f"Using completed {job}; no retraining.", flush=True)
        return row
    attempt = folder / ("attempt_" + stamp())
    attempt.mkdir()
    config = read(output / "source/configs/shared.json")
    config.update(model=name, experiment_name=name + "_matched_cuda", dataset_dir=str(data))
    save(attempt / "input_config.json", config)
    print(f"Training {job}; log: {attempt / 'train.log'}", flush=True)
    execute([sys.executable, "-u", str(output / "source/train.py"), "--config",
             str(attempt / "input_config.json"), "--output-dir", str(attempt),
             "--deterministic", "--validation-only"], attempt / "train.log")
    from protocol import split_hash
    if split_hash(read(attempt / "split_indices.json")) != read(output / "preflight.json")["dataset"]["split_sha256"]:
        raise RuntimeError("Training split differs from preflight.")
    evaluation = attempt / "final_test"
    execute([sys.executable, "-u", str(output / "source/evaluate.py"), "--attempt", str(attempt),
             "--data-dir", str(data), "--output-dir", str(evaluation)], attempt / "evaluation.log")
    training = read(attempt / "metrics.json")
    result = read(evaluation / "metrics.json")
    row = dict(model=name, threshold=0.5, metrics=result["metrics"],
               confusion_matrix=result["confusion_matrix"]["matrix"],
               best_epoch=training["best_epoch"], epochs_trained=training["epochs_trained"],
               early_stopping_triggered=training["early_stopping_triggered"],
               attempt=attempt.relative_to(output).as_posix(),
               evaluation=evaluation.relative_to(output).as_posix())
    paths = ["best_model.pt", "history.csv", "metrics.json", "state_audit.json",
             "split_indices.json", "config_snapshot.json", "validation_probs.npz",
             "final_test/metrics.json", "final_test/probabilities.npz", "final_test/predictions.csv"]
    row["artifacts"] = {(attempt / path).relative_to(output).as_posix(): file_hash(attempt / path)
                        for path in paths}
    save(completed, row)
    print(f"{job}: test accuracy={row['metrics']['accuracy']:.8f}, F1={row['metrics']['f1']:.8f}", flush=True)
    return row


def compare_repetitions(output, left, right):
    import numpy as np
    a, b = [output / row["attempt"] for row in [left, right]]
    checks = dict(history_exact=(a / "history.csv").read_bytes() == (b / "history.csv").read_bytes(),
                  state_audit_exact=read(a / "state_audit.json") == read(b / "state_audit.json"),
                  split_exact=read(a / "split_indices.json") == read(b / "split_indices.json"),
                  best_epoch_exact=left["best_epoch"] == right["best_epoch"],
                  epochs_trained_exact=left["epochs_trained"] == right["epochs_trained"],
                  test_metrics_exact=left["metrics"] == right["metrics"],
                  best_weights_exact=read(a / "final_test/metrics.json")["best_state_hash"] ==
                                     read(b / "final_test/metrics.json")["best_state_hash"])
    for name in ["validation_probs.npz", "final_test/probabilities.npz"]:
        with np.load(a / name) as x, np.load(b / name) as y:
            checks[name + "_exact"] = all(np.array_equal(x[key], y[key]) for key in ["labels", "probabilities"])
    return dict(checks=checks, all_exact=all(checks.values()))


def reference_comparison(models, expected):
    result = {}
    for name, row in models.items():
        target = expected[name]
        checks = {key: abs(row["metrics"][key] - value) < 1e-12
                  for key, value in target["metrics"].items()}
        checks.update({key: row[key] == target[key] for key in
                       ["threshold", "confusion_matrix", "best_epoch", "epochs_trained"]})
        result[name] = dict(checks=checks, matches_report=all(checks.values()))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--repeat-models", nargs="+", choices=MODELS, default=[])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--allow-environment-differences", action="store_true")
    args = parser.parse_args()
    if len(set(args.models)) != len(args.models) or not set(args.repeat_models) <= set(args.models):
        parser.error("Models must be unique; repeat models must also be selected in --models.")
    output = (args.output_dir or ROOT / "outputs" / ("run_" + stamp())).expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not args.resume:
        parser.error("Output is not empty. Use a new directory or --resume; existing results are never overwritten.")
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("This output directory is already in use.")
        def stop(signum, frame):
            raise KeyboardInterrupt(f"Received signal {signum}")
        signal.signal(signal.SIGTERM, stop)
        try:
            actual_sources = source_hashes()
            reference = read(ROOT / "reference/protocol.json")
            environment = inspect_environment()
            differences = environment_differences(environment, reference)
            save(output / "environment_check.json", dict(environment=environment, differences=differences))
            if differences and not args.allow_environment_differences:
                raise RuntimeError(f"Reference environment mismatch: {json.dumps(differences)}")
            if not args.check_only and not environment["cuda_available"]:
                raise RuntimeError("CUDA required for training, even with --allow-environment-differences.")
            if environment["gpu"] != reference["gpu"]:
                print("GPU differs from the reference Tesla T4; identical results are not guaranteed.", flush=True)
            print("Checking dataset contents and group split...", flush=True)
            data = args.data_dir.expanduser().resolve()
            dataset = inspect_dataset(data)
            validate_dataset(dataset, reference)
            manifest = dataset.pop("files")
            preflight = dict(environment=environment, environment_differences=differences,
                             dataset=dataset, source_hashes=actual_sources, models=args.models)
            if (output / "preflight.json").exists() and read(output / "preflight.json") != preflight:
                raise RuntimeError("Resume inputs/environment differ from the saved run. Start a new run.")
            save(output / "dataset_manifest.json", manifest)
            save(output / "preflight.json", preflight)
            if args.check_only:
                save(output / "status.json", dict(status="checks_passed", training_performed=False))
                print(f"Checks passed; no training/test inference performed. Details: {output}", flush=True)
                return 0
            for name in actual_sources:
                destination = output / "source" / name
                if not destination.exists():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(ROOT / name, destination)
                if file_hash(destination) != actual_sources[name]:
                    raise RuntimeError(f"Saved source changed: {name}")
            shutil.copy2(ROOT / "reference/source_hashes.json", output / "source/reference/source_hashes.json")
            (ROOT / "LAST_RUN.txt").write_text(str(output) + "\n")
            save(output / "status.json", dict(status="running"))
            models = {}
            print(f"Results: {output}", flush=True)
            for name in args.models:
                models[name] = run_model(output, name, data, name)
                save(output / "comparison.json", models)
            repetitions = {}
            for name in args.repeat_models:
                repeated = run_model(output, name, data, name + "_repeat")
                repetitions[name] = compare_repetitions(output, models[name], repeated)
                save(output / "repetition_checks.json", repetitions)
            comparison = reference_comparison(models, read(output / "source/reference/report_metrics.json"))
            save(output / "reference_comparison.json", comparison)
            execute([sys.executable, str(output / "source/plots.py"), "--run", str(output)],
                    output / ("plots_" + stamp() + ".log"))
            passed = all(row["matches_report"] for row in comparison.values())
            passed = passed and all(row["all_exact"] for row in repetitions.values())
            summary = dict(status="completed" if passed else "completed_with_mismatch", models=comparison,
                           repetitions=repetitions, all_requested_checks_passed=passed)
            save(output / "status.json", summary)
            print(json.dumps(summary, indent=2), flush=True)
            print(f"Results and PDF figures: {output}", flush=True)
            return 0 if passed else 2
        except BaseException as error:
            save(output / "status.json", dict(status="interrupted" if isinstance(error, KeyboardInterrupt)
                                              else "failed", error=str(error)))
            raise


if __name__ == "__main__":
    raise SystemExit(main())
