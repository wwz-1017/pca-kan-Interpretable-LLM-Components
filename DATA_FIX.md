# 数据集代码修复报告

## 概述

发现并修复了数据集生成脚本中的**两个严重错误**。

---

## 错误与修复

### 错误 1：缺失逗号分隔符

| 项 | 内容 |
|---|---|
| **位置** | `scripts/generate_datasets.py` → `tokenize_structured()` |
| **问题** | 最后一对与查询点粘连：`<V_0.9> <V_2.5> ?` |
| **修复** | 添加逗号分隔：`<V_0.9> , <V_2.5> ?` |
| **影响** | 破坏序列结构，模型无法学习 ICL 任务 |

**修改代码**：
```python
# 修改前
if i < len(pairs) - 1:
    ids.append(TOKEN2ID[','])

# 修改后
ids.append(TOKEN2ID[','])  # 每对都有逗号
```

---

### 错误 2：ω 参数采样错误

| 项 | 内容 |
|---|---|
| **位置** | `scripts/generate_datasets.py` → `generate_dataset()` |
| **问题** | 连续采样 `omega=6.574...` 而非离散 `{1,2,3,4,5}` |
| **修复** | 改为 `random.choice([1,2,3,4,5])` |
| **影响** | 违反规范，破坏信息瓶颈构造 |

**修改代码**：
```python
# 修改前
omega = random.uniform(0.5, 10.0)

# 修改后
OMEGA_SET = [1, 2, 3, 4, 5]
omega = random.choice(OMEGA_SET)
```

---

## 修改清单

- [x] tokenize_structured() 中移除条件，每对都加逗号
- [x] generate_dataset() 改为从 {1,2,3,4,5} 采样 ω
- [x] 删除冗余命令行参数 (--omega_dist, --omega_min, --omega_max)
- [x] 添加样本格式和 ω 参数验证

---

## 验证

修复后运行脚本应输出：
```
✓ 序列格式正确（包含逗号分隔符）
✓ ω 参数正确（离散集合 {1,2,3,4,5}）
```

---

## 后续步骤

```bash
# 1. 重新生成数据集
python3 scripts/generate_datasets.py

# 2. 验证新数据
python3 verify_data.py

# 3. 进行消融实验
python3 ablation_experiment.py
```

**状态**：✅ 代码已修改，未运行
