"""Materialize the four datasets as image folders under $ITG_DATA_ROOT (~/data).

    uv run python scripts/prepare_image_datasets.py --datasets fashion_mnist cifar10 ffhq
    uv run python scripts/prepare_image_datasets.py --datasets imagenet \
        --imagenet-source /path/to/ILSVRC2012/train

Each dataset gets `<root>/<name>/train` (the pool models draw from) and
`<root>/<name>/val`. FFHQ is the 70,000 128x128 thumbnails, split 60,000 /
10,000. ImageNet is the ILSVRC-2012 training set, one directory per class,
with the last 10 images of each class held out: 1,271,167 / 10,000.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

from itg.experiments import DATA_ROOT

FFHQ_URL = "https://huggingface.co/datasets/nuwandaa/ffhq128/resolve/main/thumbnails128x128.zip"
IMAGENET_VAL_PER_CLASS = 10
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def export_torchvision(dataset, out_dir: Path) -> int:  # noqa: ANN001 - torchvision dataset
    for index in range(len(dataset)):
        image, label = dataset[index]
        target = out_dir / f"class_{label}" / f"{index:05d}.png"
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image))
        image.convert("RGB").save(target)
    return len(dataset)


def prepare_torchvision(name: str, root: Path) -> None:
    from torchvision import datasets

    loader = {"fashion_mnist": datasets.FashionMNIST, "cifar10": datasets.CIFAR10}[name]
    for split, train in (("train", True), ("val", False)):
        count = export_torchvision(
            loader(root / "torchvision", train=train, download=True), root / name / split
        )
        print(f"{name}/{split}: {count} images")


def prepare_ffhq(root: Path) -> None:
    out = root / "ffhq"
    archive = out / "thumbnails128x128.zip"
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(["curl", "-L", "-o", str(archive), FFHQ_URL], check=True)
    with zipfile.ZipFile(archive) as handle:
        handle.extractall(out / "raw")
    paths = sorted((out / "raw").rglob("*.png"))
    assert len(paths) == 70_000, len(paths)
    for split, chunk in (("train", paths[:60_000]), ("val", paths[60_000:])):
        (out / split).mkdir(exist_ok=True)
        for path in chunk:
            path.rename(out / split / path.name)
    shutil.rmtree(out / "raw")
    archive.unlink()
    print("ffhq: 60000 train, 10000 val")


def prepare_imagenet(root: Path, source: Path) -> None:
    """Symlinks into `source`, so the 1.27M images are not copied."""
    out = root / "imagenet"
    for split in ("train", "val"):
        shutil.rmtree(out / split, ignore_errors=True)
        (out / split).mkdir(parents=True)
    counts = {"train": 0, "val": 0}
    for cls in sorted(path for path in source.iterdir() if path.is_dir()):
        files = sorted(path for path in cls.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
        split_at = len(files) - IMAGENET_VAL_PER_CLASS
        for split, chunk in (("train", files[:split_at]), ("val", files[split_at:])):
            for path in chunk:
                (out / split / f"{cls.name}_{path.name}").symlink_to(path.resolve())
            counts[split] += len(chunk)
    print(f"imagenet: {counts['train']} train, {counts['val']} val")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=None, help="default: $ITG_DATA_ROOT or ~/data")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["fashion_mnist", "cifar10", "ffhq"],
        choices=["fashion_mnist", "cifar10", "ffhq", "imagenet"],
    )
    parser.add_argument("--imagenet-source", type=Path, help="ILSVRC-2012 train, one dir per class")
    args = parser.parse_args()
    root = (args.root or DATA_ROOT).expanduser()
    for name in args.datasets:
        if name == "ffhq":
            prepare_ffhq(root)
        elif name == "imagenet":
            assert args.imagenet_source is not None, "--imagenet-source is required for imagenet"
            prepare_imagenet(root, args.imagenet_source.expanduser())
        else:
            prepare_torchvision(name, root)


if __name__ == "__main__":
    main()
