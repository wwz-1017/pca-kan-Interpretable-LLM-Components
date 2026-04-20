#!/usr/bin/env python3
"""
消融实验：hidden_dim (64 vs 128 vs 512) 对小词表/周期函数学习的影响
- 所有实验都使用 4 个注意力头
- 在验证集上快速判断收敛
- 记录训练时间、参数量、最终性能
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json
import os
import time
from datetime import datetime

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}\n")

# ==================== 模型定义 ====================

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(C, dim=-1)

        def split_heads(t):
            return t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        scale = self.d_head ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ff(self.ln2(x))
        return x


class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, dropout, max_seq_len):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def _causal_mask(self, T, device):
        mask = torch.tril(torch.ones(T, T, device=device)).unsqueeze(0).unsqueeze(0)
        return mask

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        mask = self._causal_mask(T, idx.device)

        for block in self.blocks:
            x = block(x, mask)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits


# ==================== 数据集定义 ====================

class ICLDataset(Dataset):
    def __init__(self, path, vocab_path=None):
        self.samples = []
        self.vocab_size = None

        if vocab_path and os.path.exists(vocab_path):
            try:
                with open(vocab_path, "r") as f:
                    vocab_data = json.load(f)
                self.vocab_size = len(vocab_data.get("token2id", {}))
            except Exception:
                pass

        try:
            with open(path, "r") as f:
                raw = json.load(f)
            for item in raw:
                if "tokens" in item and "label_tokens" in item:
                    if item["label_tokens"]:
                        target = int(item["label_tokens"][0])
                    else:
                        continue
                    self.samples.append({
                        "input_ids": item["tokens"],
                        "target_id": target,
                    })
        except Exception:
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except Exception:
                        continue

                    if "tokens" in item and "label_tokens" in item:
                        if item["label_tokens"]:
                            target = int(item["label_tokens"][0]) if isinstance(item["label_tokens"], list) else int(item["label_tokens"])
                        else:
                            continue
                        self.samples.append({
                            "input_ids": item["tokens"],
                            "target_id": target,
                        })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        x = torch.tensor(item["input_ids"], dtype=torch.long)
        y = torch.tensor(item["target_id"], dtype=torch.long)
        return x, y


def collate_fn(batch):
    xs, ys = zip(*batch)
    xs = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=0)
    ys = torch.stack(ys)
    return xs, ys


# ==================== 消融实验配置 ====================

ABLATION_CONFIGS = [
    {
        "name": "Small (d=64, h=4)",
        "d_model": 64,
        "n_heads": 4,
        "d_ff": 256,
    },
    {
        "name": "Medium (d=128, h=4)",
        "d_model": 128,
        "n_heads": 4,
        "d_ff": 512,
    },
    {
        "name": "Large (d=512, h=4)",
        "d_model": 512,
        "n_heads": 4,
        "d_ff": 2048,
    },
]

SHARED_CONFIG = {
    "n_layers": 2,
    "dropout": 0.1,
    "max_seq_len": 192,
    "batch_size": 256,
    "lr": 5e-4,
    "weight_decay": 0.01,
    "warmup_steps": 2000,
    "max_steps": 10000,
    "eval_every": 200,
    "patience": 5,
    "epsilon": 0.1,
    "eps_acc_threshold": 0.90,
    "mae_stable_tol": 0.02,
    "train_data": "data/train.jsonl",
    "val_data": "data/val.jsonl",
    "test_data": "data/test.jsonl",
    "vocab_path": "data/vocab.json",
}


def build_id2value(vocab_path):
    """从 vocab.json 构建 id->value 映射，用于连续指标计算。"""
    if not vocab_path or not os.path.exists(vocab_path):
        return {}
    try:
        with open(vocab_path, "r") as f:
            vocab_data = json.load(f)
        token2id = vocab_data.get("token2id", {})
        id2value = {}
        for tok, tid in token2id.items():
            if tok.startswith("<V_") and tok.endswith(">"):
                try:
                    id2value[int(tid)] = float(tok[3:-1])
                except Exception:
                    continue
        return id2value
    except Exception:
        return {}


def evaluate_continuous(model, loader, id2value, epsilon):
    """返回 exact_acc（参考）、mae、eps_acc（主指标）。"""
    exact_correct = total = 0
    abs_errors = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)[:, -1, :]
            pred = logits.argmax(dim=-1)

            exact_correct += (pred == y).sum().item()
            total += y.size(0)

            for p, t in zip(pred.cpu().numpy(), y.cpu().numpy()):
                pv = id2value.get(int(p))
                tv = id2value.get(int(t))
                if pv is not None and tv is not None:
                    abs_errors.append(abs(pv - tv))

    exact_acc = exact_correct / total if total > 0 else 0.0
    mae = float(np.mean(abs_errors)) if abs_errors else None
    eps_acc = float(np.mean(np.array(abs_errors) <= epsilon)) if abs_errors else 0.0
    return {
        "exact_acc": exact_acc,
        "mae": mae,
        "eps_acc": eps_acc,
    }

# ==================== 消融实验运行 ====================

results = {
    "timestamp": datetime.now().isoformat(),
    "device": str(DEVICE),
    "experiments": []
}

for config_idx, ablation_config in enumerate(ABLATION_CONFIGS, 1):
    print(f"\n{'='*80}")
    print(f"实验 {config_idx}/3: {ablation_config['name']}")
    print(f"{'='*80}")

    config = {**SHARED_CONFIG, **ablation_config}
    exp_name = f"d{config['d_model']}_h{config['n_heads']}"
    config["save_dir"] = f"checkpoints/ablation_{exp_name}"
    os.makedirs(config["save_dir"], exist_ok=True)
    id2value = build_id2value(config.get("vocab_path"))

    # 加载数据
    print(f"\n[加载数据]")
    train_ds = ICLDataset(config["train_data"], vocab_path=config.get("vocab_path"))
    val_ds = ICLDataset(config.get("val_data"), vocab_path=config.get("vocab_path"))
    test_ds = ICLDataset(config["test_data"], vocab_path=config.get("vocab_path"))

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"],
                              shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"],
                            shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=config["batch_size"],
                             shuffle=False, collate_fn=collate_fn)

    vocab_size = train_ds.vocab_size or 80
    print(f"  Train: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,}")

    # 创建模型
    print(f"\n[创建模型]")
    model = DecoderOnlyTransformer(
        vocab_size=vocab_size,
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        d_ff=config["d_ff"],
        dropout=config["dropout"],
        max_seq_len=config["max_seq_len"],
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {total_params:,}")

    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        betas=(0.9, 0.98),
    )

    def lr_lambda(step):
        warmup = config.get("warmup_steps", 2000)
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, config["max_steps"] - warmup)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # 训练循环
    print(f"\n[训练 (max {config['max_steps']} steps)]")

    start_time = time.time()
    step = 0
    train_iter = iter(train_loader)
    best_val_eps_acc = 0.0
    best_val_mae = float("inf")
    best_ckpt_mae = float("inf")
    stable_count = 0
    converged = False

    train_log = {
        "step": [],
        "loss": [],
        "val_exact_acc": [],
        "val_eps_acc": [],
        "val_mae": [],
        "test_exact_acc": [],
        "test_eps_acc": [],
        "test_mae": [],
    }

    model.train()

    while step < config["max_steps"] and not converged:
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)[:, -1, :]
        loss = F.cross_entropy(logits, y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        step += 1

        # 评估
        if step % config["eval_every"] == 0:
            model.eval()

            val_metrics = evaluate_continuous(model, val_loader, id2value, config.get("epsilon", 0.1))
            test_metrics = evaluate_continuous(model, test_loader, id2value, config.get("epsilon", 0.1))

            val_exact_acc = val_metrics["exact_acc"]
            val_eps_acc = val_metrics["eps_acc"]
            val_mae = val_metrics["mae"]

            test_exact_acc = test_metrics["exact_acc"]
            test_eps_acc = test_metrics["eps_acc"]
            test_mae = test_metrics["mae"]

            train_log["step"].append(step)
            train_log["loss"].append(loss.item())
            train_log["val_exact_acc"].append(val_exact_acc)
            train_log["val_eps_acc"].append(val_eps_acc)
            train_log["val_mae"].append(val_mae)
            train_log["test_exact_acc"].append(test_exact_acc)
            train_log["test_eps_acc"].append(test_eps_acc)
            train_log["test_mae"].append(test_mae)

            val_mae_str = f"{val_mae:.4f}" if val_mae is not None else "N/A"
            test_mae_str = f"{test_mae:.4f}" if test_mae is not None else "N/A"
            print(
                f"  Step {step:5d} | Loss {loss.item():.4f} | "
                f"Val ε-Acc {val_eps_acc*100:5.1f}% (MAE {val_mae_str}, Exact {val_exact_acc*100:5.1f}%) | "
                f"Test ε-Acc {test_eps_acc*100:5.1f}% (MAE {test_mae_str}, Exact {test_exact_acc*100:5.1f}%)"
            )

            if val_mae is not None:
                best_val_mae = min(best_val_mae, val_mae)

            mae_near_best = (val_mae is not None) and (val_mae <= best_val_mae + config.get("mae_stable_tol", 0.02))
            eps_good = val_eps_acc >= config.get("eps_acc_threshold", 0.90)

            if eps_good and mae_near_best:
                stable_count += 1
                if stable_count >= config["patience"]:
                    print(f"  ✓ 验证集 ε-Accuracy 连续达标且 MAE 稳定，触发早停")
                    converged = True
            else:
                stable_count = 0

            is_better = (
                (val_eps_acc > best_val_eps_acc) or
                (val_eps_acc == best_val_eps_acc and val_mae is not None and val_mae < best_ckpt_mae)
            )
            if is_better:
                best_val_eps_acc = val_eps_acc
                if val_mae is not None:
                    best_ckpt_mae = val_mae

            model.train()

    elapsed_time = time.time() - start_time

    print(f"\n[结果]")
    final_val_eps_acc = train_log["val_eps_acc"][-1] if train_log["val_eps_acc"] else 0.0
    final_test_eps_acc = train_log["test_eps_acc"][-1] if train_log["test_eps_acc"] else 0.0
    final_val_mae = train_log["val_mae"][-1] if train_log["val_mae"] else None
    final_test_mae = train_log["test_mae"][-1] if train_log["test_mae"] else None

    best_mae_str = f"{best_val_mae:.4f}" if np.isfinite(best_val_mae) else "N/A"
    final_val_mae_str = f"{final_val_mae:.4f}" if final_val_mae is not None else "N/A"
    final_test_mae_str = f"{final_test_mae:.4f}" if final_test_mae is not None else "N/A"

    print(f"  最佳验证 ε-Accuracy: {best_val_eps_acc*100:.2f}%")
    print(f"  最佳验证 MAE: {best_mae_str}")
    print(f"  最终验证 ε-Accuracy: {final_val_eps_acc*100:.2f}% | MAE: {final_val_mae_str}")
    print(f"  最终测试 ε-Accuracy: {final_test_eps_acc*100:.2f}% | MAE: {final_test_mae_str}")
    print(f"  训练时间: {elapsed_time:.1f}s")
    print(f"  实际步数: {step}/{config['max_steps']}")

    results["experiments"].append({
        "name": ablation_config['name'],
        "d_model": config["d_model"],
        "n_heads": config["n_heads"],
        "d_ff": config["d_ff"],
        "total_params": total_params,
        "training_steps": step,
        "training_time_seconds": elapsed_time,
        "best_val_eps_acc": float(best_val_eps_acc),
        "best_val_mae": None if not np.isfinite(best_val_mae) else float(best_val_mae),
        "final_val_eps_acc": float(final_val_eps_acc),
        "final_val_mae": None if final_val_mae is None else float(final_val_mae),
        "final_test_eps_acc": float(final_test_eps_acc),
        "final_test_mae": None if final_test_mae is None else float(final_test_mae),
        "converged": converged,
    })

    torch.save({
        "model_state": model.state_dict(),
        "config": config,
    }, os.path.join(config["save_dir"], "final_model.pt"))

    del model, optimizer, scheduler

# ==================== 保存并显示结果 ====================

print(f"\n{'='*80}")
print(f"消融实验完成")
print(f"{'='*80}\n")

results_path = "ablation_results.json"
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"{'模型':<30} {'参数量':<12} {'训练时间':<12} {'最佳Val ε-Acc':<16} {'最终Test ε-Acc':<16} {'收敛'}")
print("-" * 82)

for exp in results["experiments"]:
    name = exp['name']
    params = f"{exp['total_params']:,}"
    time_str = f"{exp['training_time_seconds']:.1f}s"
    best_val = f"{exp['best_val_eps_acc']*100:.1f}%"
    final_test = f"{exp['final_test_eps_acc']*100:.1f}%"
    converged = "✓" if exp['converged'] else "✗"

    print(f"{name:<30} {params:<12} {time_str:<12} {best_val:<16} {final_test:<16} {converged}")

print(f"\n结果已保存到 ablation_results.json")
