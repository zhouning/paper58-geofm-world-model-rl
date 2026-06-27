from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
SCRIPT_DIR = Path(__file__).resolve().parent
while str(SCRIPT_DIR) in sys.path:
    sys.path.remove(str(SCRIPT_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from adk_world_model.world_model import DECODER_PATH, N_CONTEXT, SCENARIOS, SCENARIO_DIM, WEIGHTS_PATH, _build_model
from scripts.rse_revision.generate_change_validation_predictions import _load_decoder


DEFAULT_EXTERNAL_EMBEDDING_DIR = ROOT / "data" / "independent_change_labels" / "embeddings"
DEFAULT_EXTERNAL_LABEL_DIR = ROOT / "data" / "independent_change_labels" / "labels"
DEFAULT_TARGET_INPUT_DIRS = [
    ROOT / "paper" / "rse_submission_paper58" / "realistic_same_grid_paper58_inputs_anxin_80m_2026-06-27",
    ROOT / "paper" / "rse_submission_paper58" / "realistic_same_grid_paper58_inputs_2026-06-27",
    ROOT / "paper" / "rse_submission_paper58" / "realistic_same_grid_paper58_inputs_dongguan_80m_2026-06-27",
    ROOT / "paper" / "rse_submission_paper58" / "realistic_same_grid_paper58_inputs_kunshan_80m_2026-06-27",
]
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "paper"
    / "rse_submission_paper58"
    / "paper58_semantic_auxiliary_latent_probe_2026-06-27"
)
EMBEDDING_RE = re.compile(r"^(?P<area>.+)_emb_(?P<year>\d{4})\.npy$")
IGNORE_INDEX = -100


@dataclass(frozen=True)
class ExternalSemanticCase:
    area: str
    start_embedding_path: Path | None
    end_embedding_path: Path | None
    context_path: Path | None
    end_label_path: Path | None
    start_label_path: Path | None = None


@dataclass(frozen=True)
class TargetSemanticCase:
    area: str
    start_year: int
    end_year: int
    start_embedding_path: Path
    context_path: Path
    start_label_path: Path
    end_label_path: Path


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _json_ready(row.get(field, "")) for field in fields})


def _is_numeric_npy(path: Path) -> bool:
    try:
        array = np.load(path, mmap_mode="r")
    except Exception:
        return False
    return getattr(array, "dtype", None) is not None and array.dtype.kind in {"b", "i", "u", "f"}


def discover_external_semantic_cases(
    embedding_dir: Path = DEFAULT_EXTERNAL_EMBEDDING_DIR,
    label_dir: Path = DEFAULT_EXTERNAL_LABEL_DIR,
    start_year: int = 2020,
    end_year: int = 2021,
) -> tuple[list[ExternalSemanticCase], list[dict[str, str]]]:
    cases: list[ExternalSemanticCase] = []
    skipped: list[dict[str, str]] = []
    embedding_root = Path(embedding_dir)
    label_root = Path(label_dir)
    for start_embedding in sorted(embedding_root.glob(f"*_emb_{int(start_year)}.npy")):
        parsed = EMBEDDING_RE.match(start_embedding.name)
        if parsed is None:
            continue
        area = str(parsed.group("area"))
        end_embedding = embedding_root / f"{area}_emb_{int(end_year)}.npy"
        context = embedding_root / f"{area}_context.npy"
        start_label = label_root / f"{area}_lulc_{int(start_year)}.npy"
        end_label = label_root / f"{area}_lulc_{int(end_year)}.npy"
        required = {
            "start_embedding": start_embedding,
            "end_embedding": end_embedding,
            "context": context,
            "start_label": start_label,
            "end_label": end_label,
        }
        missing = [name for name, path in required.items() if not path.exists()]
        if missing:
            skipped.append({"area": area, "reason": f"missing:{','.join(missing)}"})
            continue
        non_numeric = [name for name, path in required.items() if not _is_numeric_npy(path)]
        if non_numeric:
            skipped.append({"area": area, "reason": f"non_numeric_npy:{','.join(non_numeric)}"})
            continue
        cases.append(
            ExternalSemanticCase(
                area=area,
                start_embedding_path=start_embedding,
                end_embedding_path=end_embedding,
                context_path=context,
                start_label_path=start_label,
                end_label_path=end_label,
            )
        )
    return cases, skipped


def discover_target_semantic_cases(target_input_dirs: list[Path]) -> list[TargetSemanticCase]:
    cases: list[TargetSemanticCase] = []
    for root_value in target_input_dirs:
        root = Path(root_value)
        for embedding_path in sorted((root / "embeddings").glob("*_emb_*.npy")):
            parsed = EMBEDDING_RE.match(embedding_path.name)
            if parsed is None:
                continue
            area = str(parsed.group("area"))
            start_year = int(parsed.group("year"))
            end_year = start_year + 1
            context_path = root / "context" / f"{area}_context.npy"
            start_label_path = root / "labels" / f"{area}_lulc_{start_year}.npy"
            end_label_path = root / "labels" / f"{area}_lulc_{end_year}.npy"
            if context_path.exists() and start_label_path.exists() and end_label_path.exists():
                cases.append(
                    TargetSemanticCase(
                        area=area,
                        start_year=start_year,
                        end_year=end_year,
                        start_embedding_path=embedding_path,
                        context_path=context_path,
                        start_label_path=start_label_path,
                        end_label_path=end_label_path,
                    )
                )
    return cases


def filter_cases_by_terms(
    cases: list[ExternalSemanticCase],
    excluded_terms: list[str] | tuple[str, ...],
) -> list[ExternalSemanticCase]:
    terms = [str(term).lower() for term in excluded_terms if str(term).strip()]
    if not terms:
        return list(cases)
    return [case for case in cases if not any(term in case.area.lower() for term in terms)]


def split_train_calibration_cases(
    cases: list[ExternalSemanticCase],
    calibration_fraction: float,
    seed: int,
) -> tuple[list[ExternalSemanticCase], list[ExternalSemanticCase]]:
    if not cases:
        raise ValueError("no cases available for train/calibration split")
    fraction = float(calibration_fraction)
    if fraction < 0.0 or fraction >= 1.0:
        raise ValueError(f"calibration_fraction must be in [0, 1): {fraction}")
    shuffled = list(sorted(cases, key=lambda case: case.area))
    random.Random(int(seed)).shuffle(shuffled)
    n_calibration = int(round(len(shuffled) * fraction))
    if fraction > 0.0:
        n_calibration = max(1, n_calibration)
    n_calibration = min(max(0, n_calibration), max(0, len(shuffled) - 1))
    calibration = sorted(shuffled[:n_calibration], key=lambda case: case.area)
    train = sorted(shuffled[n_calibration:], key=lambda case: case.area)
    return train, calibration


def build_torch_decoder_head(decoder: Any, device: Any):
    import torch
    import torch.nn as nn

    coef = np.asarray(decoder.coef_, dtype=np.float32)
    intercept = np.asarray(decoder.intercept_, dtype=np.float32)
    if coef.ndim != 2 or intercept.ndim != 1 or coef.shape[0] != intercept.shape[0]:
        raise ValueError(f"unsupported decoder coefficient shape: coef={coef.shape}, intercept={intercept.shape}")
    head = nn.Conv2d(coef.shape[1], coef.shape[0], kernel_size=1, bias=True)
    with torch.no_grad():
        head.weight.copy_(torch.tensor(coef[:, :, None, None], dtype=torch.float32, device=device))
        head.bias.copy_(torch.tensor(intercept, dtype=torch.float32, device=device))
    head.to(device)
    head.eval()
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    return head


def labels_to_decoder_indices(
    labels: np.ndarray,
    decoder_classes: list[int] | np.ndarray,
    ignore_index: int = IGNORE_INDEX,
    device: Any | None = None,
):
    import torch

    values = np.asarray(labels)
    indices = np.full(values.shape, int(ignore_index), dtype=np.int64)
    for index, class_value in enumerate([int(value) for value in decoder_classes]):
        indices[values == class_value] = int(index)
    return torch.tensor(indices, dtype=torch.long, device=device)


def decoder_change_logits(logits: Any, start_indices: Any, ignore_index: int = IGNORE_INDEX) -> tuple[Any, Any]:
    import torch
    import torch.nn.functional as F

    if logits.ndim != 4:
        raise ValueError(f"logits must have shape [B, C, H, W], got {tuple(logits.shape)}")
    if start_indices.ndim != 3:
        raise ValueError(f"start_indices must have shape [B, H, W], got {tuple(start_indices.shape)}")
    if logits.shape[0] != start_indices.shape[0] or logits.shape[2:] != start_indices.shape[1:]:
        raise ValueError(f"shape mismatch: logits={tuple(logits.shape)}, start_indices={tuple(start_indices.shape)}")
    n_classes = int(logits.shape[1])
    valid = (start_indices != int(ignore_index)) & (start_indices >= 0) & (start_indices < n_classes)
    safe_start = torch.where(valid, start_indices, torch.zeros_like(start_indices))
    start_logit = logits.gather(1, safe_start.unsqueeze(1)).squeeze(1)
    start_one_hot = F.one_hot(safe_start, num_classes=n_classes).permute(0, 3, 1, 2).bool()
    non_start_logits = logits.masked_fill(start_one_hot, torch.finfo(logits.dtype).min)
    change_logit = torch.logsumexp(non_start_logits, dim=1) - start_logit
    change_logit = torch.where(valid, change_logit, torch.zeros_like(change_logit))
    return change_logit, valid


def semantic_allocation_loss(
    logits: Any,
    labels: Any,
    start_indices: Any | None = None,
    ignore_index: int = IGNORE_INDEX,
) -> tuple[Any, dict[str, float | int]]:
    import torch
    import torch.nn.functional as F

    if logits.ndim != 4:
        raise ValueError(f"logits must have shape [B, C, H, W], got {tuple(logits.shape)}")
    if labels.ndim != 3:
        raise ValueError(f"labels must have shape [B, H, W], got {tuple(labels.shape)}")
    if logits.shape[0] != labels.shape[0] or logits.shape[2:] != labels.shape[1:]:
        raise ValueError(f"shape mismatch: logits={tuple(logits.shape)}, labels={tuple(labels.shape)}")
    if start_indices is not None and tuple(start_indices.shape) != tuple(labels.shape):
        raise ValueError(f"start_indices shape {tuple(start_indices.shape)} does not match labels {tuple(labels.shape)}")

    n_classes = int(logits.shape[1])
    valid_labels = (labels != int(ignore_index)) & (labels >= 0) & (labels < n_classes)
    zero = torch.tensor(0.0, dtype=logits.dtype, device=logits.device)
    if not torch.any(valid_labels):
        return zero, {
            "class_quantity_l1": 0.0,
            "change_quantity_l1": 0.0,
            "valid_label_pixels": 0,
            "valid_change_pixels": 0,
        }

    probabilities = F.softmax(logits, dim=1)
    probabilities_by_pixel = probabilities.permute(0, 2, 3, 1)[valid_labels]
    true_classes = labels[valid_labels]
    true_distribution = F.one_hot(true_classes, num_classes=n_classes).to(probabilities_by_pixel.dtype).mean(dim=0)
    predicted_distribution = probabilities_by_pixel.mean(dim=0)
    class_quantity_l1 = 0.5 * torch.sum(torch.abs(predicted_distribution - true_distribution))

    change_quantity_l1 = zero
    valid_change_pixels = 0
    if start_indices is not None:
        valid_start = (start_indices != int(ignore_index)) & (start_indices >= 0) & (start_indices < n_classes)
        valid_change = valid_labels & valid_start
        valid_change_pixels = int(torch.count_nonzero(valid_change).detach().cpu())
        if torch.any(valid_change):
            safe_start = torch.where(valid_start, start_indices, torch.zeros_like(start_indices))
            start_probability = probabilities.gather(1, safe_start.unsqueeze(1)).squeeze(1)
            predicted_change_fraction = (1.0 - start_probability[valid_change]).mean()
            true_change_fraction = (labels[valid_change] != start_indices[valid_change]).to(logits.dtype).mean()
            change_quantity_l1 = torch.abs(predicted_change_fraction - true_change_fraction)

    total = class_quantity_l1 + change_quantity_l1
    return total, {
        "class_quantity_l1": float(class_quantity_l1.detach().cpu()),
        "change_quantity_l1": float(change_quantity_l1.detach().cpu()),
        "valid_label_pixels": int(torch.count_nonzero(valid_labels).detach().cpu()),
        "valid_change_pixels": valid_change_pixels,
    }


def _scenario_tensor(device: Any):
    import torch

    scenario = np.zeros(SCENARIO_DIM, dtype=np.float32)
    scenario[SCENARIOS["baseline"].id] = 1.0
    return torch.tensor(scenario, dtype=torch.float32, device=device).unsqueeze(0)


def _load_training_model(weights_path: Path, device: Any):
    import torch

    checkpoint = torch.load(Path(weights_path), map_location="cpu", weights_only=False)
    model = _build_model(
        checkpoint.get("z_dim", 64),
        checkpoint.get("scenario_dim", SCENARIO_DIM),
        checkpoint.get("n_context", N_CONTEXT),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.module.to(device)
    return model, checkpoint


def _case_tensors(case: ExternalSemanticCase, decoder_classes: list[int], device: Any) -> dict[str, Any]:
    import torch

    if (
        case.start_embedding_path is None
        or case.end_embedding_path is None
        or case.context_path is None
        or case.end_label_path is None
    ):
        raise ValueError(f"case {case.area} is missing file paths")
    start_embedding = np.load(case.start_embedding_path).astype(np.float32, copy=False)
    end_embedding = np.load(case.end_embedding_path).astype(np.float32, copy=False)
    context = np.load(case.context_path).astype(np.float32, copy=False)
    labels = np.load(case.end_label_path).astype(np.int32, copy=False)
    start_labels = (
        np.load(case.start_label_path).astype(np.int32, copy=False)
        if case.start_label_path is not None
        else np.zeros(labels.shape, dtype=np.int32)
    )
    if start_embedding.shape != end_embedding.shape:
        raise ValueError(f"embedding shape mismatch for {case.area}: {start_embedding.shape} vs {end_embedding.shape}")
    if context.shape[1:] != start_embedding.shape[:2]:
        raise ValueError(f"context shape mismatch for {case.area}: {context.shape} vs {start_embedding.shape}")
    if labels.shape != start_embedding.shape[:2]:
        raise ValueError(f"label shape mismatch for {case.area}: {labels.shape} vs {start_embedding.shape}")
    if start_labels.shape != labels.shape:
        raise ValueError(f"start label shape mismatch for {case.area}: {start_labels.shape} vs {labels.shape}")
    return {
        "z_start": torch.tensor(start_embedding.transpose(2, 0, 1), dtype=torch.float32, device=device).unsqueeze(0),
        "z_end": torch.tensor(end_embedding.transpose(2, 0, 1), dtype=torch.float32, device=device).unsqueeze(0),
        "context": torch.tensor(context, dtype=torch.float32, device=device).unsqueeze(0),
        "start_labels": labels_to_decoder_indices(start_labels, decoder_classes, device=device).unsqueeze(0),
        "labels": labels_to_decoder_indices(labels, decoder_classes, device=device).unsqueeze(0),
    }


def _class_weights_from_cases(
    cases: list[ExternalSemanticCase],
    decoder_classes: list[int],
    overrides: dict[int, float] | None,
    device: Any,
):
    import torch

    counts = np.zeros(len(decoder_classes), dtype=np.float64)
    class_to_index = {int(class_value): index for index, class_value in enumerate(decoder_classes)}
    for case in cases:
        if case.end_label_path is None:
            continue
        labels = np.load(case.end_label_path).astype(np.int32, copy=False)
        for class_value, index in class_to_index.items():
            counts[index] += int(np.count_nonzero(labels == class_value))
    positive = counts[counts > 0]
    if positive.size == 0:
        weights = np.ones(len(decoder_classes), dtype=np.float32)
    else:
        median = float(np.median(positive))
        weights = np.ones(len(decoder_classes), dtype=np.float32)
        observed = counts > 0
        weights[observed] = np.sqrt(median / np.maximum(counts[observed], 1.0)).astype(np.float32)
        weights = np.clip(weights, 0.25, 4.0)
    for class_value, factor in (overrides or {}).items():
        if int(class_value) in class_to_index:
            weights[class_to_index[int(class_value)]] *= float(factor)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _model_loss(
    model: Any,
    decoder_head: Any,
    tensors: dict[str, Any],
    scenario: Any,
    class_weights: Any,
    semantic_loss_weight: float,
    change_loss_weight: float = 0.0,
    allocation_loss_weight: float = 0.0,
    change_pos_weight: Any | None = None,
    ignore_index: int = IGNORE_INDEX,
) -> tuple[Any, dict[str, float]]:
    import torch
    import torch.nn.functional as F

    pred = model(tensors["z_start"], scenario, tensors["context"])
    pred = F.normalize(pred, p=2, dim=1)
    embedding_loss = F.mse_loss(pred, tensors["z_end"])
    logits = decoder_head(pred)
    semantic_loss = F.cross_entropy(
        logits,
        tensors["labels"],
        weight=class_weights,
        ignore_index=int(ignore_index),
    )
    change_loss = torch.tensor(0.0, dtype=embedding_loss.dtype, device=embedding_loss.device)
    allocation_loss = torch.tensor(0.0, dtype=embedding_loss.dtype, device=embedding_loss.device)
    allocation_parts: dict[str, float | int] = {
        "class_quantity_l1": 0.0,
        "change_quantity_l1": 0.0,
        "valid_label_pixels": 0,
        "valid_change_pixels": 0,
    }
    if float(change_loss_weight) > 0.0:
        change_logit, valid_start = decoder_change_logits(logits, tensors["start_labels"], ignore_index=ignore_index)
        valid_end = tensors["labels"] != int(ignore_index)
        valid_change = valid_start & valid_end
        if torch.any(valid_change):
            true_change = (tensors["labels"] != tensors["start_labels"]).float()
            change_loss = F.binary_cross_entropy_with_logits(
                change_logit[valid_change],
                true_change[valid_change],
                pos_weight=change_pos_weight,
            )
    if float(allocation_loss_weight) > 0.0:
        allocation_loss, allocation_parts = semantic_allocation_loss(
            logits,
            tensors["labels"],
            tensors["start_labels"],
            ignore_index=ignore_index,
        )
    loss = (
        embedding_loss
        + float(semantic_loss_weight) * semantic_loss
        + float(change_loss_weight) * change_loss
        + float(allocation_loss_weight) * allocation_loss
    )
    valid_labels = tensors["labels"] != int(ignore_index)
    with torch.no_grad():
        semantic_accuracy = (
            (torch.argmax(logits, dim=1)[valid_labels] == tensors["labels"][valid_labels]).float().mean().item()
            if torch.any(valid_labels)
            else 0.0
        )
    return loss, {
        "embedding_mse": float(embedding_loss.detach().cpu()),
        "semantic_ce": float(semantic_loss.detach().cpu()),
        "change_bce": float(change_loss.detach().cpu()),
        "allocation_loss": float(allocation_loss.detach().cpu()),
        "class_quantity_l1": float(allocation_parts["class_quantity_l1"]),
        "change_quantity_l1": float(allocation_parts["change_quantity_l1"]),
        "semantic_accuracy": float(semantic_accuracy),
        "total": float(loss.detach().cpu()),
    }


def _evaluate_cases(
    model: Any,
    decoder_head: Any,
    cases: list[ExternalSemanticCase],
    decoder_classes: list[int],
    scenario: Any,
    class_weights: Any,
    semantic_loss_weight: float,
    change_loss_weight: float,
    allocation_loss_weight: float,
    change_pos_weight: Any | None,
    device: Any,
) -> dict[str, float]:
    if not cases:
        return {
            "embedding_mse": 0.0,
            "semantic_ce": 0.0,
            "change_bce": 0.0,
            "allocation_loss": 0.0,
            "class_quantity_l1": 0.0,
            "change_quantity_l1": 0.0,
            "semantic_accuracy": 0.0,
            "total": 0.0,
        }
    model.eval()
    metrics: list[dict[str, float]] = []
    for case in cases:
        tensors = _case_tensors(case, decoder_classes, device)
        _, row = _model_loss(
            model,
            decoder_head,
            tensors,
            scenario,
            class_weights,
            semantic_loss_weight,
            change_loss_weight=change_loss_weight,
            allocation_loss_weight=allocation_loss_weight,
            change_pos_weight=change_pos_weight,
        )
        metrics.append(row)
    model.train(True)
    return {
        key: float(np.mean([row[key] for row in metrics]))
        for key in [
            "embedding_mse",
            "semantic_ce",
            "change_bce",
            "allocation_loss",
            "class_quantity_l1",
            "change_quantity_l1",
            "semantic_accuracy",
            "total",
        ]
    }


def _change_pos_weight_from_cases(
    cases: list[ExternalSemanticCase],
    decoder_classes: list[int],
    max_pos_weight: float,
    device: Any,
):
    import torch

    positive = 0
    negative = 0
    decoder_set = {int(value) for value in decoder_classes}
    for case in cases:
        if case.start_label_path is None or case.end_label_path is None:
            continue
        start = np.load(case.start_label_path).astype(np.int32, copy=False)
        end = np.load(case.end_label_path).astype(np.int32, copy=False)
        valid = np.isin(start, list(decoder_set)) & np.isin(end, list(decoder_set))
        positive += int(np.count_nonzero(valid & (start != end)))
        negative += int(np.count_nonzero(valid & (start == end)))
    if positive <= 0:
        value = 1.0
    else:
        value = min(float(max_pos_weight), max(1.0, float(negative / positive)))
    return torch.tensor(value, dtype=torch.float32, device=device)


def train_semantic_auxiliary_model(
    train_cases: list[ExternalSemanticCase],
    calibration_cases: list[ExternalSemanticCase],
    decoder: Any,
    output_dir: Path,
    weights_path: Path = Path(WEIGHTS_PATH),
    epochs: int = 40,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    semantic_loss_weight: float = 0.02,
    change_loss_weight: float = 0.0,
    allocation_loss_weight: float = 0.0,
    max_change_pos_weight: float = 8.0,
    class_weight_overrides: dict[int, float] | None = None,
    seed: int = 58,
    device_name: str = "cpu",
) -> tuple[Path, list[dict[str, Any]]]:
    import torch

    if not train_cases:
        raise ValueError("train_cases must not be empty")
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    random.seed(int(seed))
    device = torch.device(device_name)
    model, checkpoint = _load_training_model(Path(weights_path), device)
    decoder_classes = [int(value) for value in decoder.classes_]
    decoder_head = build_torch_decoder_head(decoder, device)
    scenario = _scenario_tensor(device)
    class_weights = _class_weights_from_cases(train_cases, decoder_classes, class_weight_overrides, device)
    change_pos_weight = _change_pos_weight_from_cases(train_cases, decoder_classes, max_change_pos_weight, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
    history: list[dict[str, Any]] = []
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_score = float("inf")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, int(epochs) + 1):
        model.train(True)
        epoch_cases = list(train_cases)
        random.Random(int(seed) + epoch).shuffle(epoch_cases)
        train_metrics: list[dict[str, float]] = []
        for case in epoch_cases:
            tensors = _case_tensors(case, decoder_classes, device)
            optimizer.zero_grad(set_to_none=True)
            loss, row = _model_loss(
                model,
                decoder_head,
                tensors,
                scenario,
                class_weights,
                semantic_loss_weight,
                change_loss_weight=change_loss_weight,
                allocation_loss_weight=allocation_loss_weight,
                change_pos_weight=change_pos_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()), max_norm=1.0)
            optimizer.step()
            train_metrics.append(row)
        train_summary = {
            key: float(np.mean([row[key] for row in train_metrics]))
            for key in [
                "embedding_mse",
                "semantic_ce",
                "change_bce",
                "allocation_loss",
                "class_quantity_l1",
                "change_quantity_l1",
                "semantic_accuracy",
                "total",
            ]
        }
        calibration_summary = _evaluate_cases(
            model,
            decoder_head,
            calibration_cases,
            decoder_classes,
            scenario,
            class_weights,
            semantic_loss_weight,
            change_loss_weight,
            allocation_loss_weight,
            change_pos_weight,
            device,
        )
        score = calibration_summary["total"] if calibration_cases else train_summary["total"]
        if score < best_score:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        history.append(
            {
                "epoch": epoch,
                "train_embedding_mse": train_summary["embedding_mse"],
                "train_semantic_ce": train_summary["semantic_ce"],
                "train_change_bce": train_summary["change_bce"],
                "train_allocation_loss": train_summary["allocation_loss"],
                "train_class_quantity_l1": train_summary["class_quantity_l1"],
                "train_change_quantity_l1": train_summary["change_quantity_l1"],
                "train_semantic_accuracy": train_summary["semantic_accuracy"],
                "train_total": train_summary["total"],
                "calibration_embedding_mse": calibration_summary["embedding_mse"],
                "calibration_semantic_ce": calibration_summary["semantic_ce"],
                "calibration_change_bce": calibration_summary["change_bce"],
                "calibration_allocation_loss": calibration_summary["allocation_loss"],
                "calibration_class_quantity_l1": calibration_summary["class_quantity_l1"],
                "calibration_change_quantity_l1": calibration_summary["change_quantity_l1"],
                "calibration_semantic_accuracy": calibration_summary["semantic_accuracy"],
                "calibration_total": calibration_summary["total"],
                "selected_score": score,
                "best_score": best_score,
            }
        )
    model.load_state_dict(best_state)
    checkpoint_path = output / "latent_dynamics_semantic_auxiliary_probe.pt"
    torch.save(
        {
            **{key: value for key, value in checkpoint.items() if key != "model_state_dict"},
            "model_state_dict": model.state_dict(),
            "semantic_auxiliary": {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "epochs": int(epochs),
                "learning_rate": float(learning_rate),
                "weight_decay": float(weight_decay),
                "semantic_loss_weight": float(semantic_loss_weight),
                "change_loss_weight": float(change_loss_weight),
                "allocation_loss_weight": float(allocation_loss_weight),
                "change_pos_weight": float(change_pos_weight.detach().cpu()),
                "decoder_classes": decoder_classes,
                "class_weight_overrides": class_weight_overrides or {},
                "best_score": best_score,
            },
        },
        checkpoint_path,
    )
    _write_csv(
        output / "training_history.csv",
        history,
        [
            "epoch",
            "train_embedding_mse",
            "train_semantic_ce",
            "train_change_bce",
            "train_allocation_loss",
            "train_class_quantity_l1",
            "train_change_quantity_l1",
            "train_semantic_accuracy",
            "train_total",
            "calibration_embedding_mse",
            "calibration_semantic_ce",
            "calibration_change_bce",
            "calibration_allocation_loss",
            "calibration_class_quantity_l1",
            "calibration_change_quantity_l1",
            "calibration_semantic_accuracy",
            "calibration_total",
            "selected_score",
            "best_score",
        ],
    )
    return checkpoint_path, history


def _predict_next_embedding(model: Any, embedding: np.ndarray, context: np.ndarray, device: Any) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    scenario = _scenario_tensor(device)
    z = torch.tensor(embedding.transpose(2, 0, 1), dtype=torch.float32, device=device).unsqueeze(0)
    context_tensor = torch.tensor(context, dtype=torch.float32, device=device).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        pred = model(z, scenario, context_tensor)
        pred = F.normalize(pred, p=2, dim=1)
    return pred.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)


def _decode_prediction(
    forecast_embedding: np.ndarray,
    decoder: Any,
    start_map: np.ndarray | None = None,
    prediction_map: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    h, w = forecast_embedding.shape[:2]
    flat = forecast_embedding.reshape(-1, forecast_embedding.shape[-1])
    prediction = decoder.predict(flat).reshape(h, w).astype(np.int32)
    score = None
    if start_map is not None:
        probabilities = decoder.predict_proba(flat).reshape(h, w, len(decoder.classes_))
        pred_for_score = prediction if prediction_map is None else np.asarray(prediction_map)
        pred_prob = np.zeros((h, w), dtype=np.float32)
        start_prob = np.zeros((h, w), dtype=np.float32)
        for column, class_value in enumerate(decoder.classes_):
            cls = int(class_value)
            pred_prob[pred_for_score == cls] = probabilities[..., column][pred_for_score == cls]
            start_prob[np.asarray(start_map) == cls] = probabilities[..., column][np.asarray(start_map) == cls]
        score = pred_prob - start_prob
    return prediction, score


def _load_model_from_checkpoint(checkpoint_path: Path, device: Any):
    import torch

    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    model = _build_model(
        checkpoint.get("z_dim", 64),
        checkpoint.get("scenario_dim", SCENARIO_DIM),
        checkpoint.get("n_context", N_CONTEXT),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.module.to(device)
    model.eval()
    return model


def _write_target_change_gate_case(
    output_dir: Path,
    case: TargetSemanticCase,
    prediction_path: Path,
    score_map: np.ndarray,
) -> Path:
    area_dir = Path(output_dir) / "target_change_gate_cases" / case.area
    diagnostics_dir = area_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    np.save(diagnostics_dir / f"{case.area}_change_gate_score.npy", score_map.astype(np.float32, copy=False))
    _write_json(
        area_dir / "manifest.json",
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "method": "paper58_semantic_auxiliary_latent_probe_score_case",
            "area": case.area,
            "start_year": case.start_year,
            "end_year": case.end_year,
            "inputs": {
                "start": case.start_label_path,
                "paper58_prediction": prediction_path,
                "embedding_start": case.start_embedding_path,
                "context": case.context_path,
            },
        },
    )
    return area_dir


def generate_semantic_auxiliary_predictions(
    checkpoint_path: Path,
    decoder: Any,
    calibration_cases: list[ExternalSemanticCase],
    target_cases: list[TargetSemanticCase],
    output_dir: Path,
    device_name: str = "cpu",
) -> dict[str, Any]:
    import torch

    device = torch.device(device_name)
    model = _load_model_from_checkpoint(Path(checkpoint_path), device)
    output = Path(output_dir)
    calibration_prediction_dir = output / "external_calibration_predictions"
    target_prediction_dir = output / "predictions"
    calibration_prediction_dir.mkdir(parents=True, exist_ok=True)
    target_prediction_dir.mkdir(parents=True, exist_ok=True)
    calibration_records: list[dict[str, Any]] = []
    for case in calibration_cases:
        if case.start_embedding_path is None or case.context_path is None:
            continue
        embedding = np.load(case.start_embedding_path).astype(np.float32, copy=False)
        context = np.load(case.context_path).astype(np.float32, copy=False)
        forecast = _predict_next_embedding(model, embedding, context, device)
        prediction, _ = _decode_prediction(forecast, decoder)
        out_path = calibration_prediction_dir / f"{case.area}_lulc_pred_2020_2021.npy"
        np.save(out_path, prediction)
        calibration_records.append({"area": case.area, "prediction": out_path})

    target_records: list[dict[str, Any]] = []
    change_gate_dirs: list[Path] = []
    for case in target_cases:
        embedding = np.load(case.start_embedding_path).astype(np.float32, copy=False)
        context = np.load(case.context_path).astype(np.float32, copy=False)
        start = np.load(case.start_label_path).astype(np.int32, copy=False)
        forecast = _predict_next_embedding(model, embedding, context, device)
        prediction, score = _decode_prediction(forecast, decoder, start_map=start)
        if score is None:
            raise RuntimeError(f"failed to compute score map for {case.area}")
        out_path = target_prediction_dir / f"{case.area}_lulc_pred_{case.start_year}_{case.end_year}.npy"
        np.save(out_path, prediction)
        change_gate_dir = _write_target_change_gate_case(output, case, out_path, score)
        change_gate_dirs.append(change_gate_dir)
        target_records.append(
            {
                "area": case.area,
                "start_year": case.start_year,
                "end_year": case.end_year,
                "prediction": out_path,
                "change_gate_dir": change_gate_dir,
                "predicted_change_pixels": int(np.count_nonzero((prediction != start) & (start != 0))),
            }
        )
    return {
        "calibration_prediction_dir": calibration_prediction_dir,
        "target_prediction_dir": target_prediction_dir,
        "target_change_gate_dirs": change_gate_dirs,
        "calibration_records": calibration_records,
        "target_records": target_records,
    }


def _parse_class_weights(values: list[str] | None) -> dict[int, float]:
    parsed: dict[int, float] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"class weight must be formatted as class=factor, got {value!r}")
        class_raw, factor_raw = value.split("=", 1)
        parsed[int(class_raw)] = float(factor_raw)
    return parsed


def run_semantic_auxiliary_probe(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    external_embedding_dir: Path = DEFAULT_EXTERNAL_EMBEDDING_DIR,
    external_label_dir: Path = DEFAULT_EXTERNAL_LABEL_DIR,
    target_input_dirs: list[Path] | None = None,
    weights_path: Path = Path(WEIGHTS_PATH),
    decoder_path: Path = Path(DECODER_PATH),
    excluded_terms: list[str] | None = None,
    calibration_fraction: float = 0.25,
    epochs: int = 40,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    semantic_loss_weight: float = 0.02,
    change_loss_weight: float = 0.0,
    allocation_loss_weight: float = 0.0,
    max_change_pos_weight: float = 8.0,
    class_weight_overrides: dict[int, float] | None = None,
    calibration_output_scope: str = "split",
    seed: int = 58,
    device_name: str = "cpu",
) -> dict[str, Any]:
    output = Path(output_dir)
    decoder = _load_decoder(Path(decoder_path))
    discovered, skipped = discover_external_semantic_cases(
        embedding_dir=Path(external_embedding_dir),
        label_dir=Path(external_label_dir),
    )
    filtered = filter_cases_by_terms(discovered, excluded_terms or [])
    train_cases, calibration_cases = split_train_calibration_cases(filtered, calibration_fraction, seed)
    target_cases = discover_target_semantic_cases(target_input_dirs or DEFAULT_TARGET_INPUT_DIRS)
    if not target_cases:
        raise ValueError("no target same-grid cases discovered")
    if calibration_output_scope not in {"split", "filtered"}:
        raise ValueError(f"calibration_output_scope must be 'split' or 'filtered': {calibration_output_scope}")
    checkpoint_path, history = train_semantic_auxiliary_model(
        train_cases=train_cases,
        calibration_cases=calibration_cases,
        decoder=decoder,
        output_dir=output,
        weights_path=Path(weights_path),
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        semantic_loss_weight=semantic_loss_weight,
        change_loss_weight=change_loss_weight,
        allocation_loss_weight=allocation_loss_weight,
        max_change_pos_weight=max_change_pos_weight,
        class_weight_overrides=class_weight_overrides,
        seed=seed,
        device_name=device_name,
    )
    prediction_manifest = generate_semantic_auxiliary_predictions(
        checkpoint_path=checkpoint_path,
        decoder=decoder,
        calibration_cases=filtered if calibration_output_scope == "filtered" else calibration_cases,
        target_cases=target_cases,
        output_dir=output,
        device_name=device_name,
    )
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "paper58_semantic_auxiliary_latent_probe",
        "output_dir": output,
        "checkpoint_path": checkpoint_path,
        "weights_path": Path(weights_path),
        "decoder_path": Path(decoder_path),
        "external_embedding_dir": Path(external_embedding_dir),
        "external_label_dir": Path(external_label_dir),
        "target_input_dirs": [Path(path) for path in (target_input_dirs or DEFAULT_TARGET_INPUT_DIRS)],
        "excluded_terms": excluded_terms or [],
        "n_discovered_external_cases": len(discovered),
        "n_filtered_external_cases": len(filtered),
        "n_train_cases": len(train_cases),
        "n_calibration_cases": len(calibration_cases),
        "skipped_external_cases": skipped,
        "train_areas": [case.area for case in train_cases],
        "calibration_areas": [case.area for case in calibration_cases],
        "target_areas": [case.area for case in target_cases],
        "parameters": {
            "calibration_fraction": float(calibration_fraction),
            "epochs": int(epochs),
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "semantic_loss_weight": float(semantic_loss_weight),
            "change_loss_weight": float(change_loss_weight),
            "allocation_loss_weight": float(allocation_loss_weight),
            "max_change_pos_weight": float(max_change_pos_weight),
            "class_weight_overrides": class_weight_overrides or {},
            "calibration_output_scope": calibration_output_scope,
            "seed": int(seed),
            "device": device_name,
        },
        "final_history_row": history[-1] if history else {},
        **prediction_manifest,
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train a strict Paper58 latent-dynamics probe with an auxiliary decoder semantic loss."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--external-embedding-dir", type=Path, default=DEFAULT_EXTERNAL_EMBEDDING_DIR)
    parser.add_argument("--external-label-dir", type=Path, default=DEFAULT_EXTERNAL_LABEL_DIR)
    parser.add_argument("--target-input-dir", type=Path, action="append", default=None)
    parser.add_argument("--weights", type=Path, default=Path(WEIGHTS_PATH))
    parser.add_argument("--decoder", type=Path, default=Path(DECODER_PATH))
    parser.add_argument("--exclude-term", action="append", default=[])
    parser.add_argument("--calibration-fraction", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--semantic-loss-weight", type=float, default=0.02)
    parser.add_argument("--change-loss-weight", type=float, default=0.0)
    parser.add_argument("--allocation-loss-weight", type=float, default=0.0)
    parser.add_argument("--max-change-pos-weight", type=float, default=8.0)
    parser.add_argument("--class-weight", action="append", default=[])
    parser.add_argument("--calibration-output-scope", choices=["split", "filtered"], default="split")
    parser.add_argument("--seed", type=int, default=58)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    manifest = run_semantic_auxiliary_probe(
        output_dir=args.output_dir,
        external_embedding_dir=args.external_embedding_dir,
        external_label_dir=args.external_label_dir,
        target_input_dirs=args.target_input_dir,
        weights_path=args.weights,
        decoder_path=args.decoder,
        excluded_terms=args.exclude_term,
        calibration_fraction=args.calibration_fraction,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        semantic_loss_weight=args.semantic_loss_weight,
        change_loss_weight=args.change_loss_weight,
        allocation_loss_weight=args.allocation_loss_weight,
        max_change_pos_weight=args.max_change_pos_weight,
        class_weight_overrides=_parse_class_weights(args.class_weight),
        calibration_output_scope=args.calibration_output_scope,
        seed=args.seed,
        device_name=args.device,
    )
    print(
        "Paper58 semantic auxiliary latent probe: "
        f"train={manifest['n_train_cases']}, "
        f"calibration={manifest['n_calibration_cases']}, "
        f"targets={len(manifest['target_records'])}, "
        f"output={manifest['output_dir']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
