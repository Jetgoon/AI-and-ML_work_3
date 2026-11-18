"""数据加载与预处理。

三件事：
1. 从 HuggingFace Datasets 拉数据
2. 切分 train / val / test（并避开"按类别排序"的坑）
3. 分词 + 转成 DataLoader
"""

import statistics
from typing import Optional, Tuple

import torch
from datasets import load_dataset
from transformers import AutoTokenizer


def build_tokenizer(cfg):
    """分词器复用官方词表。这不影响"从头训练"的结论——
    模型权重仍然随机初始化，只是"用哪套字母表"这件事没必要自己造。
    """
    return AutoTokenizer.from_pretrained(cfg.model_source())


def _take(ds, n: Optional[int], seed: int, name: str = ""):
    """先打乱再取前 n 条。

    ★ 这个顺序不能反。AG News 的原始数据是按类别排好序的，
      直接 select(range(12000)) 会拿到 3 万条 World 新闻的前 12000 条，
      模型会学到"所有输入都是 World"，准确率 25% 上不去。
      报告里"数据检查"一节可以专门写这条。
    """
    if n is None or n >= len(ds):
        return ds
    return ds.shuffle(seed=seed).select(range(n))


def load_and_split(cfg) -> Tuple:
    """返回 (train, val, test, label_names)。"""
    name, files = cfg.data_source()
    if files:
        raw = load_dataset(name, data_files=files)
    else:
        raw = load_dataset(name)
    label_names = None
    try:
        feat = raw["train"].features[cfg.label_column]
        label_names = list(feat.names)
    except Exception:
        pass

    def norm(ds):
        if cfg.label_column != "label":
            ds = ds.rename_column(cfg.label_column, "label")
        if cfg.text_column != "text":
            ds = ds.rename_column(cfg.text_column, "text")
        return ds

    train_full = norm(raw["train"])

    # val 集：有就用官方的，没有就从 train 里分层切
    if "validation" in raw and raw["validation"] is not None:
        val_ds = norm(raw["validation"])
    else:
        n_val = cfg.val_size or max(200, int(0.1 * len(train_full)))
        sp = train_full.train_test_split(
            test_size=min(0.5, max(1e-4, n_val / len(train_full))),
            seed=cfg.seed, stratify_by_column="label"
        )
        train_full, val_ds = sp["train"], sp["test"]

    test_key = "test" if "test" in raw else "validation"
    test_ds = norm(raw[test_key])

    train_ds = _take(train_full, cfg.train_size, cfg.seed, "train")
    val_ds = _take(val_ds, cfg.val_size, cfg.seed, "val")
    test_ds = _take(test_ds, cfg.test_size, cfg.seed, "test")

    return train_ds, val_ds, test_ds, label_names


def length_stats(ds, tokenizer, n: int = 2000) -> dict:
    """分词长度分布。用来给 max_length 的取值找一个有依据的理由。"""
    texts = ds["text"][: min(n, len(ds))]
    lens = [len(x) for x in tokenizer(texts)["input_ids"]]
    lens_sorted = sorted(lens)
    return {
        "mean": round(statistics.mean(lens), 1),
        "p50": lens_sorted[len(lens_sorted) // 2],
        "p95": lens_sorted[int(len(lens_sorted) * 0.95)],
        "max": lens_sorted[-1],
    }


def prepare(cfg, tokenizer):
    """分词 + 转 torch 格式。

    返回 (train, val, test, label_names, stats, test_texts)。
    test_texts 单独拎出来，因为 set_format 之后原始文本列不一定还能按列取，
    而错误分析必须拿到原文。
    """
    train_ds, val_ds, test_ds, label_names = load_and_split(cfg)

    stats = {
        "raw_size": {
            "train": len(train_ds),
            "val": len(val_ds),
            "test": len(test_ds),
        },
        "token_len": length_stats(train_ds, tokenizer),
    }

    def tokenize(batch):
        return tokenizer(
            batch["text"], truncation=True, max_length=cfg.max_length
        )

    keep = ["input_ids", "attention_mask", "labels"]
    out = []
    for idx, ds in enumerate((train_ds, val_ds, test_ds)):
        # 保留 text 列（不放进 batch，但留着做错误分析）
        drop = [c for c in ds.column_names if c not in ("text", "label", "labels")]
        ds = ds.map(tokenize, batched=True, remove_columns=drop)
        # 标签列统一改名为 labels，和 HF 模型的参数名对齐
        if "label" in ds.column_names:
            ds = ds.rename_column("label", "labels")
        if idx == 2:
            test_texts = list(ds["text"])
        ds.set_format("torch", columns=keep)
        out.append(ds)

    return out[0], out[1], out[2], label_names, stats, test_texts


def make_collator(tokenizer, pad_token_id=None):
    """自己写 padding，不用 DataCollatorWithPadding。

    为什么不用现成的：不同版本对键名（label / labels）的处理规则不一样，
    collator 一旦把不认识的键默默丢掉，你拿到的是一个没有标签的 batch，
    而报错会指向"找不到 key"这种无关痛痒的地方，很难查。

    逻辑很朴素：取这个 batch 里最长的序列，其余在右侧补 pad_token_id，
    同时生成 attention_mask（真实 token 为 1，补齐位为 0）。
    """
    pad_id = pad_token_id if pad_token_id is not None else tokenizer.pad_token_id

    def collate(features):
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, attn, labels = [], [], []
        for f in features:
            ids = f["input_ids"]
            ids = ids.tolist() if hasattr(ids, "tolist") else list(ids)
            n_pad = max_len - len(ids)
            input_ids.append(list(ids) + [pad_id] * n_pad)
            attn.append([1] * len(ids) + [0] * n_pad)
            labels.append(int(f["labels"]))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    return collate


def class_distribution(ds) -> dict:
    """类别分布，填报告 2.1 节那张表。"""
    key = "labels" if "labels" in ds.column_names else "label"
    labels = ds[key]
    if hasattr(labels, "tolist"):
        labels = labels.tolist()
    out = {}
    for y in labels:
        out[int(y)] = out.get(int(y), 0) + 1
    return dict(sorted(out.items()))
