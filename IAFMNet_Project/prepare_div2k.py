from pathlib import Path
from urllib.request import Request, urlopen
import zipfile


DATA_DIR = Path("data/div2k")

URLS = {
    "DIV2K_train_HR.zip": (
        "https://data.vision.ee.ethz.ch/cvl/DIV2K/"
        "DIV2K_train_HR.zip"
    ),
    "DIV2K_train_LR_bicubic_X4.zip": (
        "https://data.vision.ee.ethz.ch/cvl/DIV2K/"
        "DIV2K_train_LR_bicubic_X4.zip"
    ),
}


def download_file(url: str, output_path: Path) -> None:
    if output_path.exists():
        print(f"Already exists: {output_path.name}")
        return

    print(f"Downloading: {output_path.name}")

    request = Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"},
    )

    with urlopen(request, timeout=300) as response:
        total = int(
            response.headers.get("Content-Length", 0)
        )
        downloaded = 0

        with output_path.open("wb") as file:
            while True:
                chunk = response.read(1024 * 1024)

                if not chunk:
                    break

                file.write(chunk)
                downloaded += len(chunk)

                if total:
                    percent = downloaded * 100 // total
                    print(
                        f"\rProgress: {percent}%",
                        end="",
                    )

    print("\nDownload complete.")


def extract_zip(zip_path: Path, output_dir: Path) -> None:
    print(f"Extracting: {zip_path.name}")

    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(output_dir)

    print("Extraction complete.")


def count_pngs(directory: Path) -> int:
    if not directory.exists():
        return 0

    return len(list(directory.glob("*.png")))


def prepare_dataset() -> None:
    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for filename, url in URLS.items():
        zip_path = DATA_DIR / filename

        download_file(url, zip_path)
        extract_zip(zip_path, DATA_DIR)

    hr_dir = DATA_DIR / "DIV2K_train_HR"
    lr_dir = DATA_DIR / "DIV2K_train_LR_bicubic" / "X4"

    hr_count = count_pngs(hr_dir)
    lr_count = count_pngs(lr_dir)

    print()
    print("=" * 50)
    print("DIV2K DATASET CHECK")
    print("=" * 50)
    print(f"HR images: {hr_count}")
    print(f"LR images: {lr_count}")

    if hr_count == 0:
        raise RuntimeError(
            f"No HR images found in: {hr_dir}"
        )

    if lr_count == 0:
        raise RuntimeError(
            f"No LR images found in: {lr_dir}"
        )

    if hr_count != lr_count:
        raise RuntimeError(
            "HR and LR image counts do not match."
        )

    print()
    print("DIV2K dataset is ready for training.")


if __name__ == "__main__":
    prepare_dataset()