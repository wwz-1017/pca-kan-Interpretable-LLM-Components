"""
Minimal Decoder-only Transformer for ICL Function Learning
架构: 4层, 512维, 用于PCA+KAN可解释性pipeline的受控实验台
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
import json

# ── 0. 全局配置 ──────────────────────────────────────────────────────────────
CONFIG = {
    # 模型结构
    "d_model":      512,
    "n_heads":      8,       # 512 / 8 = 64维每头，标准配置
    "n_layers":     4,       # 从 2 增加到 4（更深的模型）
    "d_ff":         2048,    # FFN隐层 = 4 * d_model，标准比例
    "dropout":      0.1,
    "max_seq_len":  192,     # 适配 q=25（约154 token），避免位置编码越界

    # 训练
    "batch_size":   256,
    "lr":           5e-4,    # 从 1e-3 降低到 5e-4（更稳定的学习率）
    "weight_decay": 0.01,    # 减弱正则化
    "max_steps":    10000,   # 缩短为 10000 步（快速实验）
    "eval_every":   200,     # 每200步评估一次
    "patience":     5,       # ε-Accuracy 连续满足阈值次数
    "epsilon":      0.1,     # ε-邻域半径（与量化步长一致）
    "eps_acc_threshold": 0.90,
    "mae_stable_tol": 0.02,  # 认为“接近最优MAE”的容忍区间
    
    # Warmup 调度
    "warmup_steps": 2000,    # 增加到 2000 步

    # 数据路径（使用新生成的三个分离数据集）
    "train_data":   "data/train.jsonl",
    "val_data":     "data/val.jsonl",
    "test_data":    "data/test.jsonl",
    "vocab_path":   "data/vocab.json",

    # 保存
    "save_dir":     "checkpoints",
    "log_path":     "training_log.json",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ── 1. 数据集 ─────────────────────────────────────────────────────────────────
class ICLDataset(Dataset):
    """
    期望数据格式（来自 icl_sin_q5.jsonl 的 JSONL 每行格式）:
    每行包含类似字段: {"omega": int, "text": str, "tokens": [...], "label_tokens": [...], ...}

    本类会把每行转换为内部统一格式:
      {"input_ids": tokens, "target_id": last_token_of_label_tokens, "omega": omega}

    特性:
      - 使用 JSONL（一行一个 JSON 对象）或 JSON list（向后兼容）。
      - 使用字段 "tokens" 作为输入序列。
      - 使用 "label_tokens" 的最后一个 token 作为单步预测目标（整数 id）。
      - 自动加载 vocab.json 验证词表大小（如果存在）。
      - 若 label_tokens 为空或缺失，则该样本会被跳过。
    """
    def __init__(self, path, vocab_path=None):
        self.samples = []
        self.vocab_size = None
        
        # 尝试加载 vocab.json 获取精确的词表大小
        if vocab_path and os.path.exists(vocab_path):
            try:
                with open(vocab_path, "r") as f:
                    vocab_data = json.load(f)
                self.vocab_size = len(vocab_data.get("token2id", {}))
                print(f"✓ 已加载词表（vocab_size={self.vocab_size}）")
            except Exception as e:
                print(f"⚠ 加载词表失败: {e}，将从数据推断")
        
        # 支持 JSON list 或 JSONL（每行一个 JSON）
        try:
            # 先尝试整个文件作为 JSON 列表加载（向后兼容）
            with open(path, "r") as f:
                raw = json.load(f)
            for item in raw:
                if "tokens" in item and "label_tokens" in item:
                    if item["label_tokens"]:
                        # 新格式：label_tokens = [target_id, <EOS>_id]，取第一个
                        target = int(item["label_tokens"][0])
                    else:
                        continue
                    self.samples.append({
                        "input_ids": item["tokens"],
                        "target_id": target,
                        "omega": item.get("omega", 1)
                    })
                elif "input_ids" in item and "target_id" in item:
                    # 已经是旧格式，直接使用
                    self.samples.append({
                        "input_ids": item["input_ids"],
                        "target_id": int(item["target_id"]),
                        "omega": item.get("omega", 1)
                    })
        except Exception:
            # 如果不是 JSON 列表，再按 JSONL 每行解析
            with open(path, "r") as f:
                for lineno, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except Exception:
                        # 解析失败，跳过并继续
                        print(f"Warning: failed to parse JSON on line {lineno} in {path}")
                        continue

                    # 期望字段为 tokens 和 label_tokens
                    if "tokens" in item and "label_tokens" in item:
                        if item["label_tokens"]:
                            # 新格式：label_tokens 可能是 [target_id, <EOS>_id]
                            # 或者旧格式的单个值，都兼容
                            target = int(item["label_tokens"][0]) if isinstance(item["label_tokens"], list) else int(item["label_tokens"])
                        else:
                            continue
                        self.samples.append({
                            "input_ids": item["tokens"],
                            "target_id": target,
                            "omega": item.get("omega", 1)
                        })
                    elif "input_ids" in item and "target_id" in item:
                        self.samples.append({
                            "input_ids": item["input_ids"],
                            "target_id": int(item["target_id"]),
                            "omega": item.get("omega", 1)
                        })
                    else:
                        # 不支持的行格式，跳过
                        continue

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        x = torch.tensor(item["input_ids"], dtype=torch.long)
        y = torch.tensor(item["target_id"],  dtype=torch.long)
        return x, y


def collate_fn(batch):
    """等长padding"""
    xs, ys = zip(*batch)
    xs = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=0)
    ys = torch.stack(ys)
    return xs, ys


# ── 2. 模型定义 ───────────────────────────────────────────────────────────────
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj  = nn.Linear(d_model, d_model,     bias=False)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        qkv = self.qkv_proj(x)                          # (B, T, 3C)
        q, k, v = qkv.split(C, dim=-1)                  # 各(B, T, C)

        # 拆分多头
        def split_heads(t):
            return t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        # Scaled dot-product attention
        scale  = self.d_head ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, T, T)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn   = F.softmax(scores, dim=-1)
        attn   = self.dropout(attn)
        out    = torch.matmul(attn, v)                  # (B, H, T, d_head)

        # 合并多头
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2  = nn.LayerNorm(d_model)
        self.ff   = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, mask=None):
        # Pre-LN结构（比Post-LN更稳定）
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ff(self.ln2(x))
        return x


class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff,
                 dropout, max_seq_len):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop    = nn.Dropout(dropout)

        self.blocks  = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.ln_f    = nn.LayerNorm(d_model)
        self.head    = nn.Linear(d_model, vocab_size, bias=False)

        # 权重绑定（token embedding与输出head共享权重，标准做法）
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
        """因果掩码：每个位置只能看到自己和之前的token"""
        mask = torch.tril(torch.ones(T, T, device=device)).unsqueeze(0).unsqueeze(0)
        return mask  # (1, 1, T, T)

    def forward(self, idx):
        B, T = idx.shape
        assert T <= CONFIG["max_seq_len"], f"序列长度{T}超过max_seq_len{CONFIG['max_seq_len']}"

        pos  = torch.arange(T, device=idx.device).unsqueeze(0)  # (1, T)
        x    = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        mask = self._causal_mask(T, idx.device)

        for block in self.blocks:
            x = block(x, mask)

        x    = self.ln_f(x)
        logits = self.head(x)                # (B, T, vocab_size)
        return logits

    def get_layer_activations(self, idx, layer_idx):
        """
        ── Pipeline专用接口 ──
        提取指定层TransformerBlock输出后、进入下一层之前的激活值
        用于Day 4的PCA+KAN分析
        layer_idx: 0 或 1（对应第1、2层）
        返回: (B, T, d_model) 的激活张量
        """
        B, T = idx.shape
        pos  = torch.arange(T, device=idx.device).unsqueeze(0)
        x    = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        mask = self._causal_mask(T, idx.device)

        for i, block in enumerate(self.blocks):
            x = block(x, mask)
            if i == layer_idx:
                return x.detach()  # (B, T, 512) ← 这就是你要喂给PCA的东西

        return x.detach()


# ── 3. 训练循环 ───────────────────────────────────────────────────────────────

def build_id2value(token2id):
    """将 token2id 构建为 id->数值 的查找表，仅保留 <V_*> token。"""
    if token2id is None:
        return {}

    id2value = {}
    for tok, tid in token2id.items():
        if tok.startswith('<V_') and tok.endswith('>'):
            try:
                id2value[int(tid)] = float(tok[3:-1])
            except Exception:
                continue
    return id2value


def token_to_value(token_id, id2value=None):
    """
    将 token ID 转换回数值
    
    Y 值对应的 token 是 <V_-1.0> 到 <V_1.0>，ID 范围约 [8, 28]
    需要从 vocab.json 反推
    """
    if id2value is None:
        return None
    return id2value.get(int(token_id), None)


def evaluate(model, loader, device, id2value=None, epsilon=0.1):
    """
    连续域评估：
    - MAE: 数值平均绝对误差
    - ε-Accuracy: |ŷ - y| <= ε 的比例
    说明：exact token accuracy 仅作参考，不作为主指标。
    """
    model.eval()
    exact_correct = total = 0
    abs_errors = []
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)               # (B, T, vocab)
            last   = logits[:, -1, :]       # (B, vocab) 取最后位置
            pred   = last.argmax(dim=-1)    # (B,)
            
            # exact match（仅记录，不作为主指标）
            exact_correct += (pred == y).sum().item()
            total   += y.size(0)
            
            # 连续域指标
            if id2value is not None:
                for p, t in zip(pred.cpu().numpy(), y.cpu().numpy()):
                    pred_val = token_to_value(int(p), id2value)
                    true_val = token_to_value(int(t), id2value)
                    if pred_val is not None and true_val is not None:
                        abs_errors.append(abs(pred_val - true_val))
    
    model.train()
    
    exact_acc = exact_correct / total if total > 0 else 0.0
    mae = float(np.mean(abs_errors)) if abs_errors else None
    eps_acc = float(np.mean(np.array(abs_errors) <= epsilon)) if abs_errors else 0.0

    return {
        "exact_acc": exact_acc,
        "mae": mae,
        "eps_acc": eps_acc,
        "n_numeric": len(abs_errors),
    }


def train(config):
    os.makedirs(config["save_dir"], exist_ok=True)

    # 数据
    train_ds = ICLDataset(config["train_data"], vocab_path=config.get("vocab_path"))
    val_ds   = ICLDataset(config.get("val_data", config["test_data"]), vocab_path=config.get("vocab_path"))
    test_ds  = ICLDataset(config["test_data"],  vocab_path=config.get("vocab_path"))
    
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"],
                              shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,  batch_size=config["batch_size"],
                              shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=config["batch_size"],
                              shuffle=False, collate_fn=collate_fn)

    # 获取词表大小：优先使用 vocab.json，否则从数据推断
    if train_ds.vocab_size is not None:
        vocab_size = train_ds.vocab_size
        print(f"✓ 使用词表大小: {vocab_size}")
    else:
        # 向后兼容：从数据推断
        all_ids = [sid for item in train_ds.samples for sid in item["input_ids"]]
        all_ids += [item["target_id"] for item in train_ds.samples]
        vocab_size = max(all_ids) + 1
        print(f"⚠ 从数据推断词表大小: {vocab_size}")
    
    print(f"训练集: {len(train_ds):,} 条 | 验证集: {len(val_ds):,} 条 | 测试集: {len(test_ds):,} 条")

    # 模型
    model = DecoderOnlyTransformer(
        vocab_size   = vocab_size,
        d_model      = config["d_model"],
        n_heads      = config["n_heads"],
        n_layers     = config["n_layers"],
        d_ff         = config["d_ff"],
        dropout      = config["dropout"],
        max_seq_len  = config["max_seq_len"],
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    # 优化器：AdamW（lr=5e-4, weight_decay=0.01）
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = config["lr"],
        weight_decay = config["weight_decay"],
        betas        = (0.9, 0.98),
    )

    # 学习率调度：线性warmup后cosine衰减
    def lr_lambda(step):
        warmup = config.get("warmup_steps", 2000)  # 从 CONFIG 读取，默认 2000
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, config["max_steps"] - warmup)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # 加载 token2id 构建数值映射（用于 MAE / ε-Accuracy）
    token2id = None
    id2value = None
    try:
        vocab_path = config.get("vocab_path", "data/vocab.json")
        with open(vocab_path, "r") as f:
            vocab_data = json.load(f)
            if isinstance(vocab_data, dict) and "token2id" in vocab_data:
                token2id = vocab_data["token2id"]
                id2value = build_id2value(token2id)
                print(f"✓ 加载 token2id ({len(token2id)} 个 token), 可解析数值token={len(id2value)}")
    except Exception as e:
        print(f"⚠ 无法加载 token2id（{e}），连续域指标将被跳过")
    
    # 训练记录
    log = {
        "step": [], "train_loss": [], 
        "val_exact_acc": [], "val_eps_acc": [], "val_mae": [],
        "test_exact_acc": [], "test_eps_acc": [], "test_mae": [],
        "lr": []
    }
    best_eps_acc    = 0.0
    best_val_mae    = float("inf")
    best_ckpt_mae   = float("inf")
    stable_count    = 0
    converged     = False

    step         = 0
    train_iter   = iter(train_loader)

    print("\n开始训练，等待grokking发生...\n")

    while step < config["max_steps"] and not converged:
        # 无限循环数据
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(DEVICE), y.to(DEVICE)

        # 前向传播：取最后位置的logit计算loss
        logits = model(x)[:, -1, :]   # (B, vocab)
        loss   = F.cross_entropy(logits, y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        step += 1

        # 定期评估
        if step % config["eval_every"] == 0:
            val_metrics = evaluate(
                model, val_loader, DEVICE,
                id2value=id2value,
                epsilon=config.get("epsilon", 0.1)
            )
            test_metrics = evaluate(
                model, test_loader, DEVICE,
                id2value=id2value,
                epsilon=config.get("epsilon", 0.1)
            )
            lr_now   = scheduler.get_last_lr()[0]

            val_exact_acc = val_metrics["exact_acc"]
            val_eps_acc = val_metrics["eps_acc"]
            val_mae = val_metrics["mae"]

            test_exact_acc = test_metrics["exact_acc"]
            test_eps_acc = test_metrics["eps_acc"]
            test_mae = test_metrics["mae"]

            log["step"].append(step)
            log["train_loss"].append(loss.item())
            log["val_exact_acc"].append(val_exact_acc)
            log["val_eps_acc"].append(val_eps_acc)
            log["val_mae"].append(val_mae)
            log["test_exact_acc"].append(test_exact_acc)
            log["test_eps_acc"].append(test_eps_acc)
            log["test_mae"].append(test_mae)
            log["lr"].append(lr_now)

            # 格式化输出
            val_mae_str = f"{val_mae:.4f}" if val_mae is not None else "N/A"
            test_mae_str = f"{test_mae:.4f}" if test_mae is not None else "N/A"
            
            print(f"Step {step:6d} | Loss {loss.item():.4f} | "
                  f"Val ε-Acc {val_eps_acc*100:.1f}% (MAE {val_mae_str}, Exact {val_exact_acc*100:.1f}%) | "
                  f"Test ε-Acc {test_eps_acc*100:.1f}% (MAE {test_mae_str}, Exact {test_exact_acc*100:.1f}%) | LR {lr_now:.2e}")

            # 收敛检测（连续域）：ε-Accuracy 连续达标 + MAE 接近最优区间
            if val_mae is not None:
                best_val_mae = min(best_val_mae, val_mae)

            mae_near_best = (val_mae is not None) and (val_mae <= best_val_mae + config.get("mae_stable_tol", 0.02))
            eps_good = val_eps_acc >= config.get("eps_acc_threshold", 0.90)

            if eps_good and mae_near_best:
                stable_count += 1
                if stable_count >= config["patience"]:
                    print(
                        f"\n✓ Grokking确认！验证集 ε-Accuracy 连续{config['patience']}次"
                        f"≥{config.get('eps_acc_threshold', 0.90)*100:.0f}% 且 MAE 收敛于最优区间"
                    )
                    converged = True
            else:
                stable_count = 0  # 一旦掉下去就重置

            # 保存最优模型（主指标: ε-Accuracy，次指标: MAE）
            is_better = (
                (val_eps_acc > best_eps_acc) or
                (val_eps_acc == best_eps_acc and val_mae is not None and val_mae < best_ckpt_mae)
            )
            if is_better:
                best_eps_acc = val_eps_acc
                if val_mae is not None:
                    best_ckpt_mae = val_mae
                ckpt_path = os.path.join(config["save_dir"], "best_model.pt")
                torch.save({
                    "step":        step,
                    "model_state": model.state_dict(),
                    "config":      config,
                    "vocab_size":  vocab_size,
                    "val_eps_acc": val_eps_acc,
                    "val_mae":     val_mae,
                    "test_eps_acc": test_eps_acc,
                    "test_mae":    test_mae,
                }, ckpt_path)
                print(f"  → 已保存最优模型 (val ε-Acc={best_eps_acc*100:.1f}%, val MAE={val_mae_str})")

    # 保存训练日志
    with open(config["log_path"], "w") as f:
        json.dump(log, f, indent=2)

    # 绘制训练曲线
    plot_training_curve(log)

    best_mae_str = f"{best_val_mae:.4f}" if np.isfinite(best_val_mae) else "N/A"
    print(f"\n训练结束 | 最优验证 ε-Accuracy: {best_eps_acc*100:.1f}% | 最优验证 MAE: {best_mae_str}")
    return model, log


def plot_training_curve(log):
    steps = log.get("step", [])

    # 兼容旧日志字段
    train_loss = log.get("train_loss", [])
    val_eps_acc = log.get("val_eps_acc", [])
    test_eps_acc = log.get("test_eps_acc", log.get("test_acc", []))
    val_mae = log.get("val_mae", [])
    test_mae = log.get("test_mae", [])

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    # 1) Train Loss
    ax1.plot(steps, train_loss, color="steelblue", linewidth=1.5, label="Train Loss")
    ax1.set_ylabel("Train Loss")
    ax1.set_yscale("log")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)

    # 2) ε-Accuracy
    if val_eps_acc:
        ax2.plot(steps, [a * 100 for a in val_eps_acc], color="seagreen", linewidth=1.5, label="Val ε-Acc")
    if test_eps_acc:
        ax2.plot(steps, [a * 100 for a in test_eps_acc], color="darkorange", linewidth=1.5, label="Test ε-Acc")
    eps_thr = CONFIG.get("eps_acc_threshold", 0.90) * 100
    ax2.axhline(eps_thr, color="red", linestyle="--", alpha=0.7, label=f"Epsilon threshold {eps_thr:.0f}%")
    ax2.set_ylabel("ε-Accuracy (%)")
    ax2.set_ylim(0, 102)
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)

    # 3) MAE（将 None 显示为 NaN，保持曲线连续语义）
    if val_mae:
        val_mae_plot = [np.nan if v is None else v for v in val_mae]
        ax3.plot(steps, val_mae_plot, color="purple", linewidth=1.5, label="Val MAE")
    if test_mae:
        test_mae_plot = [np.nan if v is None else v for v in test_mae]
        ax3.plot(steps, test_mae_plot, color="brown", linewidth=1.5, label="Test MAE")
    ax3.set_ylabel("MAE")
    ax3.set_xlabel("Training Steps")
    ax3.legend(loc="best")
    ax3.grid(True, alpha=0.3)

    plt.suptitle("Training Curves (Loss / Epsilon-Accuracy / MAE)", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig("training_curve.png", dpi=150)
    print("训练曲线已保存为 training_curve.png")


# ── 4. 激活提取接口（Day 4 Pipeline用）─────────────────────────────────────────
def extract_activations(model_path, data_path, layer_idx=1, position=-1):
    """
    Day 4 PCA+KAN pipeline的入口函数
    
    参数:
        model_path: 训练好的模型checkpoint路径
        data_path:  要提取激活的数据集路径
        layer_idx:  提取哪一层的激活（0=第1层, 1=第2层）
        position:   提取序列哪个位置的激活（-1=最后一个token，即query位置）
    
    返回:
        activations: numpy array (N, 512)  ← 喂给PCA
        targets:     numpy array (N,)      ← 喂给KAN作为回归目标
    """
    # 加载checkpoint
    ckpt       = torch.load(model_path, map_location=DEVICE)
    vocab_size = ckpt["vocab_size"]
    cfg        = ckpt["config"]

    model = DecoderOnlyTransformer(
        vocab_size   = vocab_size,
        d_model      = cfg["d_model"],
        n_heads      = cfg["n_heads"],
        n_layers     = cfg["n_layers"],
        d_ff         = cfg["d_ff"],
        dropout      = 0.0,              # 提取时关掉dropout
        max_seq_len  = cfg["max_seq_len"],
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    dataset = ICLDataset(data_path)
    loader  = DataLoader(dataset, batch_size=256,
                         shuffle=False, collate_fn=collate_fn)

    all_acts    = []
    all_targets = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            acts = model.get_layer_activations(x, layer_idx)  # (B, T, 512)
            # 取指定位置
            acts = acts[:, position, :]                        # (B, 512)
            all_acts.append(acts.cpu().numpy())
            all_targets.append(y.numpy())

    activations = np.concatenate(all_acts,    axis=0)  # (N, 512)
    targets     = np.concatenate(all_targets, axis=0)  # (N,)

    print(f"激活提取完成: shape={activations.shape}, 目标shape={targets.shape}")
    np.save("activations.npy", activations)
    np.save("targets.npy",     targets)
    print("已保存为 activations.npy 和 targets.npy")

    return activations, targets


# ── 5. 主入口 ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 训练
    model, log = train(CONFIG)

    # 训练完成后，提取第2层激活（用于pipeline）
    print("\n提取激活供PCA+KAN使用...")
    extract_activations(
        model_path = os.path.join(CONFIG["save_dir"], "best_model.pt"),
        data_path  = CONFIG["test_data"],
        layer_idx  = 1,    # 第2层（最后一层）
        position   = -1,   # query token位置
    )