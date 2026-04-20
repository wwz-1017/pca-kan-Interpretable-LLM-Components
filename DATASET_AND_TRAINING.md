# ICL 正弦数据集 & Transformer 训练框架 - 完整文档

## 📊 数据集配置

### 数据规模
- **训练集**: 100,000 条样本
- **验证集**: 10,000 条样本
- **测试集**: 10,000 条样本
- **总计**: 120,000 条样本

### ω 参数分布
- **分布方式**: 均匀分布（Uniform）
- **范围**: [0.5, 10.0]
- **采样方式**: 同分布采样（训练、验证、测试集分别使用不同的随机种子）

### 上下文长度
- **Context 对数 (q)**: 5 个示例对
- **典型序列长度**: ~30-35 个 token

---

## 📝 数据格式

### JSONL 行格式
```json
{
  "omega": 6.574554585349896,
  "text": "<BOS><V_-3.0> <V_-0.7> , <V_-1.4> <V_-0.1> , ...",
  "tokens": [1, 8, 5, 31, 5, 6, 5, 24, ...],
  "label_text": "<V_-0.5><EOS>",
  "label_tokens": [33, 2]
}
```

### 字段说明
| 字段 | 类型 | 说明 |
|------|------|------|
| `omega` | float | 正弦函数的角频率 (ω ∈ [0.5, 10.0]) |
| `text` | str | 人类可读的序列（用于调试） |
| `tokens` | int[] | Token ID 序列 |
| `label_text` | str | 目标值和 EOS 符的可读形式 |
| `label_tokens` | int[] | [target_id, eos_id] 其中 target_id 是 Y 值对应的 token |

---

## 🔤 词表结构（Vocab Size = 80）

### 分布明细
| 类型 | Token | 数量 | ID范围 |
|------|-------|------|--------|
| 控制符 | `<PAD>`, `<BOS>`, `<EOS>`, `<UNK>` | 4 | 0-3 |
| 特殊符 | ` ` (空格), `,` (逗号), `?` (查询) | 3 | 4-6 |
| 数值符 | `<V_-3.1>` 到 `<V_3.1>` (步长0.1) | 63 | 7-69 |
| 填充符 | `<PAD_0>` 到 `<PAD_9>` | 10 | 70-79 |

### 关键特性
- **词表大小 80**: 是 8 的倍数，优化 CUDA 矩阵运算
- **统一数值词表**: X 和 Y 共享同一套 63 个数值 token（因为 Y ⊂ X）
- **包含空格 token**: 空格作为独立的 token（ID=5），分隔序列元素

---

## 📐 序列格式

### Prompt 格式（输入序列）
```
<BOS> <V_x1> <SP> <V_y1> <SP> , <SP> <V_x2> <SP> <V_y2> <SP> , ... <V_xq> <SP> ?
```

### Label 格式（目标）
```
<V_yq> <EOS>
```

### 具体例子
**Prompt**: `<BOS>` `<V_-3.0>` `<SP>` `<V_-0.7>` `<SP>` `,` `<SP>` ... `<V_2.5>` `<SP>` `?`

**Label**: `<V_-0.5>` `<EOS>`

---

## 🏋️ 训练框架更新

### CONFIG 配置
```python
CONFIG = {
    "train_data":   "data/train.jsonl",
    "val_data":     "data/val.jsonl",
    "test_data":    "data/test.jsonl",
    "vocab_path":   "data/vocab.json",
    # ... 其他配置保持不变
}
```

### ICLDataset 类改进
- ✅ 自动加载 `vocab.json` 获取精确的 vocab_size
- ✅ 支持新的 label_tokens 格式 `[target_id, eos_id]`
- ✅ 自动提取第一个元素作为目标 ID
- ✅ 向后兼容旧格式

### 训练循环改进
- ✅ 支持验证集评估（每 `eval_every` 步）
- ✅ 基于验证集判断收敛（而非测试集）
- ✅ 保存时记录 val_acc 和 test_acc 两个指标
- ✅ 日志输出同时显示验证集和测试集准确率

---

## 📂 文件清单

### 生成脚本
- **`scripts/generate_datasets.py`** - 生成三个数据集（train/val/test）
  ```bash
  python3 scripts/generate_datasets.py \
    --train_n 100000 --val_n 10000 --test_n 10000 \
    --q 5 --omega_dist uniform --omega_min 0.5 --omega_max 10.0
  ```

### 训练脚本
- **`scripts/train transfomer.py`** - 主训练框架
  ```bash
  python3 scripts/train\ transfomer.py
  ```

### 验证脚本
- **`verify_data.py`** - 验证数据加载
  ```bash
  python3 verify_data.py
  ```

### 数据文件
- **`data/train.jsonl`** - 100,000 条训练样本
- **`data/val.jsonl`** - 10,000 条验证样本
- **`data/test.jsonl`** - 10,000 条测试样本
- **`data/vocab.json`** - 词表映射 (token2id, id2token)

---

## ✅ 验证清单

- [x] 数据集正确生成（100k+10k+10k）
- [x] ω 参数采用同分布采样
- [x] 词表大小 = 80（CUDA 优化）
- [x] 所有 token ID 在 [0, 79] 范围内
- [x] Label tokens 格式正确 `[target_id, eos_id]`
- [x] ICLDataset 正确加载所有三个数据集
- [x] DataLoader 正确处理批处理和填充
- [x] train_transformer.py 支持验证集

---

## 🚀 使用指南

### 第一步：生成数据（如需重新生成）
```bash
cd /Users/a1/projects/pca-kan
python3 scripts/generate_datasets.py
```

### 第二步：验证数据加载
```bash
python3 verify_data.py
```

### 第三步：开始训练
```bash
python3 scripts/train\ transfomer.py
```

### 输出日志示例
```
[验证数据加载]
✓ 已加载词表（vocab_size=80）
✓ 训练集: 100,000 条
✓ 验证集: 10,000 条
✓ 测试集: 10,000 条

[开始训练，等待grokking发生...]
Step    200 | Loss 4.3824 | Val Acc 12.5% | Test Acc 11.8% | LR 9.90e-04
Step    400 | Loss 3.8901 | Val Acc 25.3% | Test Acc 24.9% | LR 9.80e-04
...
✓ Grokking确认！验证准确率连续5次≥95%
训练结束 | 最优验证准确率: 98.5%
```

---

## 📈 关键改进

1. **完整的三集分离**: train/val/test 独立，便于监测 generalization
2. **同分布采样**: 三个集合都从同一 ω 分布采样，确保公平对比
3. **改进的评估**: 使用验证集判断收敛，避免过度拟合测试集
4. **清晰的日志**: 每个 eval_every 步骤同时显示 val 和 test 准确率
5. **CUDA 优化**: 词表大小 80 = 8×10，矩阵运算效率最优

---

## 🔍 数据质量指标

```
✓ 输入张量 shape: torch.Size([32, 32])  (batch_size=32, max_seq_len)
✓ 目标张量 shape: torch.Size([32])      (batch_size=32)
✓ 输入范围: [1, 69]                      (合法 token ID)
✓ 目标范围: [28, 48]                     (Y 值对应的 token ID)
✓ 所有 token ID 都在 [0, 79] 范围内     (词表内有效)
```

---

*Last Updated: 2026-04-16*
