#!/usr/bin/env python3
"""
将 icl_sin_q5.jsonl 分割为训练集和测试集（80% / 20%）
"""
import json
import argparse
import os

def split_jsonl(input_path, train_path, test_path, train_ratio=0.8, seed=42):
    """
    分割 JSONL 数据集
    
    参数:
        input_path: 输入 JSONL 文件路径
        train_path: 输出训练集 JSONL 路径
        test_path: 输出测试集 JSONL 路径
        train_ratio: 训练集比例（默认 0.8）
        seed: 随机种子
    """
    import random
    random.seed(seed)
    
    # 读取所有行
    samples = []
    with open(input_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    
    print(f"总共读取 {len(samples)} 个样本")
    
    # 随机分割
    random.shuffle(samples)
    split_idx = int(len(samples) * train_ratio)
    train_samples = samples[:split_idx]
    test_samples = samples[split_idx:]
    
    # 保存训练集
    os.makedirs(os.path.dirname(train_path) or '.', exist_ok=True)
    with open(train_path, 'w') as f:
        for sample in train_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    # 保存测试集
    os.makedirs(os.path.dirname(test_path) or '.', exist_ok=True)
    with open(test_path, 'w') as f:
        for sample in test_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"✓ 训练集: {len(train_samples)} 个样本 → {train_path}")
    print(f"✓ 测试集: {len(test_samples)} 个样本 → {test_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split JSONL dataset into train/test')
    parser.add_argument('--input', type=str, default='data/icl_sin_q5.jsonl',
                        help='Input JSONL file')
    parser.add_argument('--train', type=str, default='data/train.jsonl',
                        help='Output train JSONL file')
    parser.add_argument('--test', type=str, default='data/test.jsonl',
                        help='Output test JSONL file')
    parser.add_argument('--ratio', type=float, default=0.8,
                        help='Training set ratio (default 0.8)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    split_jsonl(args.input, args.train, args.test, 
                train_ratio=args.ratio, seed=args.seed)
