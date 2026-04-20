#!/usr/bin/env python3
"""
生成三个分离的 ICL 正弦数据集：训练集、验证集、测试集
- 训练集: 100,000 条
- 验证集: 10,000 条
- 测试集: 10,000 条
- ω 参数: 从区间 [1.0, 5.0] 连续均匀采样
- φ 参数: 从区间 [0, 2π) 连续均匀采样
"""
import json
import math
import random
from typing import List
import argparse
import os

# ========== 1. 词表精确定义 ==========

# 控制符（4 个）
CONTROL_TOKENS = ['<PAD>', '<BOS>', '<EOS>', '<UNK>']

# 特殊符号（3 个）
SPECIAL_TOKENS = [' ', ',', '?']

# 数值范围 [-3.1, 3.1]，步长 0.1 -> 63 个离散值
X_VALUES = [round(i * 0.1, 1) for i in range(-31, 32)]

# 构建数值 Token
VALUE_TOKENS = [f"<V_{val:.1f}>" for val in X_VALUES]

# 组装基础词表
TOKEN_LIST = CONTROL_TOKENS + SPECIAL_TOKENS + VALUE_TOKENS  # 4 + 3 + 63 = 70 个

# 填充至 80（为了 CUDA 矩阵运算效率）
PADDING_TOKENS = [f"<PAD_{i}>" for i in range(80 - len(TOKEN_LIST))]
TOKEN_LIST = TOKEN_LIST + PADDING_TOKENS

# 映射字典
TOKEN2ID = {tok: i for i, tok in enumerate(TOKEN_LIST)}
ID2TOKEN = {i: tok for tok, i in TOKEN2ID.items()}
VOCAB_SIZE = len(TOKEN_LIST)

print(f"[词表初始化]")
print(f"  控制符: {len(CONTROL_TOKENS)} 个")
print(f"  特殊符: {len(SPECIAL_TOKENS)} 个")
print(f"  数值符: {len(VALUE_TOKENS)} 个")
print(f"  填充符: {len(PADDING_TOKENS)} 个")
print(f"  -> 总词表大小: {VOCAB_SIZE}")


def get_value_token(val: float) -> str:
    """获取数值对应的 Token，严格处理 -0.0"""
    val_rounded = round(val, 1)
    if val_rounded == -0.0:
        val_rounded = 0.0
    
    tok = f"<V_{val_rounded:.1f}>"
    if tok not in TOKEN2ID:
        raise ValueError(f"数值 {val} (舍入为 {val_rounded}) 超出词表预设范围 [-3.1, 3.1]")
    return tok


def tokenize_structured(pairs: List[tuple], xq: float) -> List[int]:
    """
    严格按照格式生成 Prompt 的 Token ID 序列:
    <BOS> <V_x1> <SP> <V_y1> <SP> , <SP> <V_x2> <SP> <V_y2> <SP> , <SP> ... <SP> , <SP> <V_xq> <SP> ?
    
    注意：最后一对 (xq, yq) 前必须有 ", " 分隔符，确保结构清晰
    """
    ids = [TOKEN2ID['<BOS>']]  # 起始符
    
    for i, (x, y) in enumerate(pairs):
        ids.append(TOKEN2ID[get_value_token(x)])
        ids.append(TOKEN2ID[' '])
        
        ids.append(TOKEN2ID[get_value_token(y)])
        
        # 每一对后面都有 ", " 分隔（包括最后一对，因为后面还有查询点 xq）
        ids.append(TOKEN2ID[' '])  # y 之后的空格
        ids.append(TOKEN2ID[','])  # 逗号分隔
        ids.append(TOKEN2ID[' '])  # 逗号之后的空格
            
    # 查询点 xq
    ids.append(TOKEN2ID[get_value_token(xq)])
    ids.append(TOKEN2ID[' '])
    ids.append(TOKEN2ID['?'])
    
    return ids


def decode_tokens(ids: List[int]) -> str:
    """将 token ID 精确还原为字符串"""
    return "".join([ID2TOKEN[idx] for idx in ids])


def make_example(omega: float, phi: float, q: int) -> dict:
    """生成单条 ICL 样本数据"""
    pairs = []
    for _ in range(q):
        x = random.uniform(-math.pi, math.pi)
        y = math.sin(omega * x + phi)
        pairs.append((x, y))
    
    xq = random.uniform(-math.pi, math.pi)
    yq = math.sin(omega * xq + phi)
    
    # 生成 Prompt 的 Tokens 和严格对应的文本
    prompt_tokens = tokenize_structured(pairs, xq)
    prompt_text = decode_tokens(prompt_tokens)
    
    # 生成 Label（目标值 + EOS 结束符）
    yq_tok = get_value_token(yq)
    label_tokens = [TOKEN2ID[yq_tok], TOKEN2ID['<EOS>']]
    label_text = decode_tokens(label_tokens)
    
    return {
        "omega": omega,
        "phi": phi,
        "text": prompt_text,
        "tokens": prompt_tokens,
        "label_text": label_text,
        "label_tokens": label_tokens,
    }


def generate_dataset(n: int, q: int = 25, seed: int = None):
    """
    生成数据集
    
    参数:
        n: 样本数量
        q: 上下文示例对数
        seed: 随机种子
    
    注意：
        ω 从 [1.0, 5.0] 连续均匀采样
        φ 从 [0, 2π) 连续均匀采样
    """
    if seed is not None:
        random.seed(seed)
    
    out = []
    for idx in range(n):
        omega = round(random.uniform(1.0, 5.0), 2)
        phi = round(random.uniform(0.0, 2 * math.pi), 2)
        out.append(make_example(omega, phi, q))
        
        if (idx + 1) % 10000 == 0:
            print(f"    已生成 {idx + 1}/{n} 条样本...")
    
    return out


def save_dataset(ds: List[dict], output_path: str):
    """保存数据集为 JSONL"""
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for ex in ds:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def save_vocab(output_dir: str):
    """保存词表"""
    vocab_path = os.path.join(output_dir, 'vocab.json')
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump({
            "token2id": TOKEN2ID,
            "id2token": ID2TOKEN
        }, f, ensure_ascii=False, indent=2)
    return vocab_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='生成训练、验证、测试三个 ICL 正弦数据集（ω∈[1,5], φ∈[0,2π) 连续采样）'
    )
    parser.add_argument('--train_n', type=int, default=100000, help='训练集样本数')
    parser.add_argument('--val_n', type=int, default=10000, help='验证集样本数')
    parser.add_argument('--test_n', type=int, default=10000, help='测试集样本数')
    parser.add_argument('--q', type=int, default=25, help='上下文示例对数')
    parser.add_argument('--out_dir', type=str, default='data', help='输出目录')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    args = parser.parse_args()
    
    print(f"\n[数据集生成配置]")
    print(f"  训练集: {args.train_n:,} 条样本")
    print(f"  验证集: {args.val_n:,} 条样本")
    print(f"  测试集: {args.test_n:,} 条样本")
    print(f"  上下文长度: {args.q} 个示例对")
    print(f"  ω 采样范围: [1.0, 5.0] (连续均匀分布)")
    print(f"  φ 采样范围: [0, 2π) (连续均匀分布)")
    print(f"  随机种子: {args.seed}")
    
    # 生成三个数据集
    print(f"\n[生成训练集]")
    train_ds = generate_dataset(args.train_n, args.q, seed=args.seed)
    
    print(f"\n[生成验证集]")
    val_ds = generate_dataset(args.val_n, args.q, seed=args.seed + 1)
    
    print(f"\n[生成测试集]")
    test_ds = generate_dataset(args.test_n, args.q, seed=args.seed + 2)
    
    # 保存数据集
    print(f"\n[保存数据集到 {args.out_dir}]")
    os.makedirs(args.out_dir, exist_ok=True)
    
    train_path = os.path.join(args.out_dir, 'train.jsonl')
    val_path = os.path.join(args.out_dir, 'val.jsonl')
    test_path = os.path.join(args.out_dir, 'test.jsonl')
    
    save_dataset(train_ds, train_path)
    print(f"  ✓ 训练集: {train_path} ({len(train_ds):,} 条)")
    
    save_dataset(val_ds, val_path)
    print(f"  ✓ 验证集: {val_path} ({len(val_ds):,} 条)")
    
    save_dataset(test_ds, test_path)
    print(f"  ✓ 测试集: {test_path} ({len(test_ds):,} 条)")
    
    # 保存词表
    vocab_path = save_vocab(args.out_dir)
    print(f"  ✓ 词表: {vocab_path}")
    
    # 打印样本示例
    print(f"\n[样本示例 - 格式验证]")
    sample = train_ds[0]
    print(f"  Omega      : {sample['omega']:.2f} (应在 [1.0, 5.0])")
    print(f"  Phi        : {sample['phi']:.2f} (应在 [0, 2π))")
    print(f"  Prompt 文本: {sample['text']}")
    print(f"  Label 文本 : {sample['label_text']}")
    print(f"  Token 数   : {len(sample['tokens'])} 个")
    
    # 格式检查
    text = sample['text']
    if ', ' in text and text.endswith('?'):
        print(f"  ✓ 序列格式正确（包含逗号分隔符，以问号结尾）")
    else:
        print(f"  ✗ 序列格式错误！")
    
    if 1.0 <= sample['omega'] <= 5.0:
        print(f"  ✓ ω 参数正确（连续区间 [1.0, 5.0]）")
    else:
        print(f"  ✗ ω 参数错误！应在 [1.0, 5.0]")

    if 0.0 <= sample['phi'] < 2 * math.pi:
        print(f"  ✓ φ 参数正确（连续区间 [0, 2π)）")
    else:
        print(f"  ✗ φ 参数错误！应在 [0, 2π)")
    
    print(f"\n✅ 数据集生成完成！")
