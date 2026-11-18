"""模型构建。

这里只做一件事，但这件事决定了你的作业算不算"从头训练"：
用 AutoConfig 取架构配置，用 from_config 随机初始化权重，
而不是 from_pretrained 把别人的权重直接搬过来。
"""

import math

import torch
from transformers import AutoConfig, AutoModelForSequenceClassification


def build_model(cfg):
    src = cfg.model_source()
    if cfg.use_pretrained_weights:
        model = AutoModelForSequenceClassification.from_pretrained(
            src, num_labels=cfg.num_labels
        )
        mode = "pretrained"
    else:
        # 只读 config.json：hidden_size / num_layers / num_heads 全在里面
        conf = AutoConfig.from_pretrained(src)
        conf.num_labels = cfg.num_labels
        # 权重按 initializer_range 随机初始化，一个预训练参数都没加载
        model = AutoModelForSequenceClassification.from_config(conf)
        mode = "scratch"
    return model, mode


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def param_breakdown(model):
    """按模块拆参数量，填报告 3.3 节那张表。"""
    buckets = {}
    for name, p in model.named_parameters():
        if "embedding" in name.lower():
            k = "Embedding"
        elif "classifier" in name or "pre_classifier" in name:
            k = "Classifier"
        elif "pooler" in name:
            k = "Pooler"
        else:
            k = "Encoder"
        buckets[k] = buckets.get(k, 0) + p.numel()
    total = sum(buckets.values())
    return {
        "detail": {k: {"params": v, "ratio": round(v / total * 100, 2)}
                   for k, v in sorted(buckets.items())},
        "total": total,
    }


def arch_summary(model):
    """提取架构超参，填报告 3.2 节。"""
    c = model.config
    n_layer = getattr(c, "n_layers", None) or getattr(c, "num_hidden_layers", None)
    n_head = getattr(c, "n_heads", None) or getattr(c, "num_attention_heads", None)
    hidden = getattr(c, "dim", None) or getattr(c, "hidden_size", None)
    ffn = getattr(c, "hidden_dim", None) or getattr(c, "intermediate_size", None)
    return {
        "arch": type(model).__name__,
        "hidden_size": hidden,
        "num_layers": n_layer,
        "num_heads": n_head,
        "ffn_dim": ffn,
        "activation": getattr(c, "hidden_act", getattr(c, "activation", None)),
        "vocab_size": getattr(c, "vocab_size", None),
        "max_position": getattr(c, "max_position_embeddings", None),
        "dropout": getattr(c, "dropout", getattr(c, "hidden_dropout_prob", None)),
        "num_labels": getattr(c, "num_labels", None),
    }


def verify_random_init(model, num_labels):
    """证明权重确实没被预训练过。

    随机初始化时 embedding 权重服从 N(0, initializer_range)，
    数值都在 ±0.05 以内且均值接近 0；类别数 k 的理论初始 loss 是 ln(k)。

    注意别去看第 0 行——那是 [PAD] 的 embedding，BERT 类模型会把 padding_idx
    整行置零，永远是 0.0，拿它当证据会得出错误结论。这里取第 1 行（[UNK]）。
    """
    emb = model.get_input_embeddings().weight.detach()
    row = emb[1]
    head = [round(float(v), 4) for v in row[:5]]
    theoretical = round(math.log(num_labels), 4)
    return {
        "embedding_row_index": 1,
        "embedding_first_row_head": head,
        "embedding_abs_mean": round(float(emb[1:].abs().mean()), 5),
        "theoretical_init_loss": theoretical,
    }
