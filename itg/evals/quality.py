"""clean-FID scoring: from sample tensors, or a directory of generated PNGs."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from itg.evals.images import as_images

CleanInception = tuple[object, torch.nn.Module, torch.device]


@lru_cache(maxsize=1)
def _clean_inception() -> CleanInception:
    from cleanfid.fid import build_feature_extractor, build_resizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_feature_extractor("clean", device, use_dataparallel=False)
    return build_resizer("clean"), model, device


def clean_features(samples: Tensor, *, batch_size: int = 64) -> np.ndarray:
    """clean-FID Inception features, via cleanfid's own resizer and network."""
    from cleanfid.fid import get_batch_features
    from torchvision.transforms import ToTensor

    resize, model, device = _clean_inception()
    to_tensor = ToTensor()

    def prepare(image: np.ndarray) -> torch.Tensor:
        # cleanfid's resizer returns float32 already in [0, 255]; ToTensor only
        # rescales uint8. Multiplying unconditionally is 255x too bright.
        resized = resize(image)
        tensor = to_tensor(resized)
        return tensor * 255 if resized.dtype == "uint8" else tensor

    images = as_images(samples)
    chunks = [
        get_batch_features(
            torch.stack([prepare(image) for image in images[start : start + batch_size]]),
            model,
            device,
        )
        for start in range(0, len(images), batch_size)
    ]
    return np.concatenate(chunks)


def clean_fid(generated: np.ndarray, reference: np.ndarray) -> float:
    from cleanfid.fid import fid_from_feats

    return float(fid_from_feats(generated, reference))


def score_images(
    image_dir: Path,
    output_dir: Path,
    *,
    reference_dir: Path,
    dataset_name: str = "cifar10",
    resolution: int = 32,
    device: str = "cuda",
    force: bool = False,
) -> dict[str, float | int | str]:
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists() and not force:
        print(f"[metrics] cached: {metrics_path}", flush=True)
        payload: dict[str, float | int | str] = json.loads(metrics_path.read_text())
        return payload

    import cleanfid
    from cleanfid import fid

    images = sorted(image_dir.rglob("*.png"))
    if not images:
        raise ValueError(f"no PNG images under {image_dir}")
    common = {
        "fdir1": str(image_dir),
        "dataset_name": dataset_name,
        "dataset_res": resolution,
        "dataset_split": "train",
        "mode": "clean",
        "batch_size": 64,
        "num_workers": 0,
        "device": device,
        "use_dataparallel": False,
    }
    payload = {"image_dir": str(image_dir), "num_images": len(images)}
    try:
        payload["clean_fid"] = float(fid.compute_fid(**common))
    except Exception as error:
        payload["clean_fid_error"] = repr(error)
        print(f"[metrics] clean_fid failed: {error!r}", flush=True)

    local_name = f"{dataset_name}_local"
    kid_stats = Path(cleanfid.__file__).parent / "stats" / f"{local_name}_clean_custom_na_kid.npz"
    try:
        if not kid_stats.exists():
            fid.make_custom_stats(
                local_name,
                str(reference_dir.expanduser().resolve()),
                mode="clean",
                num_workers=0,
                batch_size=64,
                device=device,
            )
        payload["kid"] = float(
            fid.compute_kid(**{**common, "dataset_name": local_name, "dataset_split": "custom"})
        )
    except Exception as error:
        payload["kid_error"] = repr(error)
        print(f"[metrics] kid failed: {error!r}", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = metrics_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(metrics_path)
    print(f"[metrics] {json.dumps(payload, sort_keys=True)}", flush=True)
    return payload
