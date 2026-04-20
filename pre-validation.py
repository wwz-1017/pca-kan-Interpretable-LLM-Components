#!/usr/bin/env python3
"""
Pre-validation before full training:
1) Use only 10% training data (10,000 samples)
2) Train for 10,000 steps
3) Check acceptance criteria:
   - val MAE can drop below 0.15 within 10,000 steps
   - |train MAE - val MAE| <= 0.05
"""

import importlib.util
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
TRAIN_SCRIPT_PATH = ROOT / "scripts" / "train transfomer_512.py"
DATA_DIR = ROOT / "data"
SUBSET_PATH = DATA_DIR / "train_10pct_10k.jsonl"
RUN_LOG_PATH = ROOT / "pre_validation_log.json"
SUMMARY_PATH = ROOT / "pre_validation_summary.json"

N_TRAIN_SUBSET = 10_000
MAX_STEPS = 10_000

VAL_MAE_THRESHOLD = 0.15
MAE_GAP_THRESHOLD = 0.05


def import_train_module(script_path: Path):
    spec = importlib.util.spec_from_file_location("train_mod", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_subset(src_path: Path, dst_path: Path, n_lines: int):
    """Write first n_lines from source JSONL to destination JSONL."""
    count = 0
    with src_path.open("r", encoding="utf-8") as fin, dst_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            fout.write(line)
            count += 1
            if count >= n_lines:
                break

    if count < n_lines:
        raise ValueError(f"Not enough samples in {src_path}, got {count}, need {n_lines}")

    return count


def min_non_none(values):
    vals = [v for v in values if v is not None]
    return min(vals) if vals else None


def main():
    train_mod = import_train_module(TRAIN_SCRIPT_PATH)

    # 1) Build 10% subset
    src_train = DATA_DIR / "train.jsonl"
    subset_count = build_subset(src_train, SUBSET_PATH, N_TRAIN_SUBSET)
    print(f"Built subset: {SUBSET_PATH} ({subset_count} samples)")

    # 2) Configure run
    config = dict(train_mod.CONFIG)
    config.update(
        {
            "train_data": str(SUBSET_PATH),
            "val_data": str(DATA_DIR / "val.jsonl"),
            "test_data": str(DATA_DIR / "test.jsonl"),
            "vocab_path": str(DATA_DIR / "vocab.json"),
            "max_steps": MAX_STEPS,
            "save_dir": str(ROOT / "checkpoints" / "pre_validation_10pct"),
            "log_path": str(RUN_LOG_PATH),
        }
    )

    print("\n=== Pre-validation run config ===")
    print(json.dumps({
        "train_data": config["train_data"],
        "val_data": config["val_data"],
        "max_steps": config["max_steps"],
        "batch_size": config["batch_size"],
        "lr": config["lr"],
        "weight_decay": config["weight_decay"],
        "warmup_steps": config.get("warmup_steps", 2000),
        "epsilon": config.get("epsilon", 0.1),
        "eps_acc_threshold": config.get("eps_acc_threshold", 0.9),
    }, indent=2))

    # 3) Train
    model, log = train_mod.train(config)

    # 4) Post-train MAE evaluation on train/val
    with open(config["vocab_path"], "r", encoding="utf-8") as f:
        vocab_data = json.load(f)
    token2id = vocab_data["token2id"]
    id2value = train_mod.build_id2value(token2id)

    train_ds = train_mod.ICLDataset(config["train_data"], vocab_path=config.get("vocab_path"))
    val_ds = train_mod.ICLDataset(config["val_data"], vocab_path=config.get("vocab_path"))

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=False, collate_fn=train_mod.collate_fn)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, collate_fn=train_mod.collate_fn)

    train_metrics = train_mod.evaluate(
        model,
        train_loader,
        train_mod.DEVICE,
        id2value=id2value,
        epsilon=config.get("epsilon", 0.1),
    )
    val_metrics = train_mod.evaluate(
        model,
        val_loader,
        train_mod.DEVICE,
        id2value=id2value,
        epsilon=config.get("epsilon", 0.1),
    )

    best_val_mae = min_non_none(log.get("val_mae", []))
    train_mae = train_metrics.get("mae")
    val_mae = val_metrics.get("mae")
    mae_gap = None if (train_mae is None or val_mae is None) else abs(train_mae - val_mae)

    criterion_1 = (best_val_mae is not None) and (best_val_mae < VAL_MAE_THRESHOLD)
    criterion_2 = (mae_gap is not None) and (mae_gap <= MAE_GAP_THRESHOLD)
    passed = criterion_1 and criterion_2

    summary = {
        "subset_train_samples": subset_count,
        "max_steps": MAX_STEPS,
        "best_val_mae_within_10k": best_val_mae,
        "final_train_mae": train_mae,
        "final_val_mae": val_mae,
        "final_mae_gap": mae_gap,
        "criterion_1_best_val_mae_lt_0_15": criterion_1,
        "criterion_2_train_val_gap_le_0_05": criterion_2,
        "pass": passed,
    }

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Pre-validation Summary ===")
    print(json.dumps(summary, indent=2))

    if passed:
        print("\nRESULT: PASS")
        raise SystemExit(0)
    else:
        print("\nRESULT: FAIL")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
