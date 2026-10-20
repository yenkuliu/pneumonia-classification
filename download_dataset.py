"""Download the Kaggle dataset, then record the directory containing train/test."""

from pathlib import Path


def find_dataset(download):
    candidates = [download, *sorted(download.rglob("chest_xray"))]
    for candidate in candidates:
        if all((candidate / split / label).is_dir()
               for split in ["train", "test"] for label in ["NORMAL", "PNEUMONIA"]):
            return candidate.resolve()
    raise FileNotFoundError(f"No complete chest_xray folder in {download}")


def main():
    import kagglehub
    download = Path(kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia"))
    dataset = find_dataset(download)
    pointer = Path(__file__).resolve().parent / "dataset_path.txt"
    pointer.write_text(str(dataset) + "\n")
    print(f"Dataset: {dataset}")
    print("The reproduction preflight will verify contents against the reference dataset checksum.")


if __name__ == "__main__":
    main()
