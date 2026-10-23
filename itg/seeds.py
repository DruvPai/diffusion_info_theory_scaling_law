"""Pure seed derivations and explicit RNG.

No ambient RNG state anywhere in the package: every random op takes a
generator built here from a `stable_seed`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch


def stable_seed(base_seed: int, *parts: object) -> int:
    """sha256 truncation of `(base_seed, *parts)` into [0, 2**31)."""
    raw = "|".join([str(base_seed), *(str(part) for part in parts)]).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], "big") % (2**31)


def generator(seed: int, device: str | torch.device = "cpu") -> torch.Generator:
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    return g


def draw_seed(g: torch.Generator) -> int:
    """One int drawn off `g`, for seeding a device-side generator."""
    return int(torch.randint(0, 2**31, (), generator=g).item())


@dataclass(frozen=True)
class CaseSeeds:
    """Every seed one case uses, as a pure function of (base_seed, case_name)."""

    base_seed: int
    case_name: str

    @property
    def case_eval_seed(self) -> int:
        return stable_seed(self.base_seed, self.case_name, "case_eval")

    @property
    def immse_seed(self) -> int:
        return stable_seed(self.base_seed, self.case_name, "immse")

    def sample_seed(self, role: str) -> int:
        return stable_seed(self.base_seed, self.case_name, "samples", role)

    def model_dataset_seed(self, model_idx: int) -> int:
        return stable_seed(self.base_seed, self.case_name, model_idx, "dataset")

    def model_train_seed(self, model_idx: int) -> int:
        return stable_seed(self.base_seed, self.case_name, model_idx, "train")

    def aux_network_train_seed(self, name: str, k: int) -> int:
        """Seed for the size-k network of kind `name` ("mixture"/"classifier")."""
        return stable_seed(self.base_seed, self.case_name, name, k, "train")
