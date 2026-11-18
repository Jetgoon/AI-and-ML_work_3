"""训练主脚本：一个完整的手写训练循环。

刻意不用 HuggingFace Trainer，两个原因：
1. 作业要求"从头到尾"，每一步都该看得见，Trainer 帮你藏了太多
2. Trainer 的 API 每个大版本都在改，手写循环反而不会因为版本升级跑不起来
"""

import argparse
import csv
import json
import math
import os
import random
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from torch.utils.data import DataLoader

import config as cfgmod
import data as datamod
import model as modelmod

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# --------------------------------------------------------------------------
# 基础工具
# --------------------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def split_batch(batch, device):
    """把 batch 拆成模型输入和标签。标签键名是 labels，和 HF 模型参数名一致。"""
    labels = batch["labels"].to(device)
    inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
    return inputs, labels


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, n = 0.0, 0
    all_preds, all_labels, all_probs = [], [], []
    for batch in loader:
        inputs, labels = split_batch(batch, device)
        out = model(**inputs)
        loss = F.cross_entropy(out.logits, labels, reduction="sum")
        total_loss += float(loss)
        n += labels.size(0)
        probs = F.softmax(out.logits, dim=-1)
        all_probs.append(probs.detach().cpu())
        all_preds.append(out.logits.argmax(-1).detach().cpu())
        all_labels.append(labels.cpu())

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()
    probs = torch.cat(all_probs).numpy()
    return {
        "loss": total_loss / max(n, 1),
        "preds": y_pred,
        "labels": y_true,
        "probs": probs,
        "acc": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
    }


# --------------------------------------------------------------------------
# 画图（图里只用英文，避免 matplotlib 中文字体缺失显示成方块）
# --------------------------------------------------------------------------

def plot_step_loss(step_losses, out_path, log_every=50):
    if not step_losses:
        return
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    xs = list(range(1, len(step_losses) + 1))
    ax.plot(xs, step_losses, color="#B5D4F4", linewidth=0.8, label="per-step loss")
    w = max(5, len(step_losses) // 40)
    if len(step_losses) > w:
        kernel = np.ones(w) / w
        smooth = np.convolve(step_losses, kernel, mode="valid")
        ax.plot(range(w, len(step_losses) + 1), smooth,
                color="#185FA5", linewidth=1.8, label="moving average (w={})".format(w))
    ax.set_xlabel("Training step")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Training loss")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_epoch_curve(history, out_path):
    ep = [h["epoch"] for h in history]
    fig, ax1 = plt.subplots(figsize=(6.4, 3.8))
    ax1.plot(ep, [h["train_loss"] for h in history], "o-", color="#185FA5", label="train loss")
    ax1.plot(ep, [h["val_loss"] for h in history], "s-", color="#D85A30", label="val loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_xticks(ep)
    if len(ep) == 1:
        ax1.set_xlim(ep[0] - 0.5, ep[0] + 0.5)
    ax2 = ax1.twinx()
    ax2.plot(ep, [h["val_acc"] for h in history], "^--", color="#5F5E5A",
             label="val accuracy")
    ax2.set_ylabel("Validation accuracy")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, fontsize=9, loc="center right")
    ax1.set_title("Loss and accuracy per epoch")
    ax1.spines[["top"]].set_visible(False)
    ax2.spines[["top"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_confusion(cm, names, out_path):
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    k = cm.shape[0]
    ax.set_xticks(range(k))
    ax.set_yticks(range(k))
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (test set)")
    thr = cm.max() / 2.0 if cm.max() else 0
    for i in range(k):
        for j in range(k):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=9,
                    color="white" if cm[i, j] > thr else "#2C2C2A")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def dump_errors(texts, y_true, y_pred, probs, names, out_path, topk=300):
    """导出错分样本，按模型置信度从高到低排。

    "错得最自信"的样本是最有价值的人工分析对象——
    模型在那儿摔得最狠，说明那儿是知识边界。
    """
    rows = []
    for t, yt, yp, pr in zip(texts, y_true, y_pred, probs):
        if int(yt) != int(yp):
            rows.append({
                "text": str(t).replace("\n", " ")[:300],
                "true": names[int(yt)] if names else int(yt),
                "pred": names[int(yp)] if names else int(yp),
                "confidence": round(float(pr[int(yp)]), 4),
            })
    rows.sort(key=lambda r: -r["confidence"])
    rows = rows[:topk]
    if not rows:
        return 0
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["text", "true", "pred", "confidence"])
        w.writeheader()
        w.writerows(rows)
    return len(rows)


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------

def run(cfg):
    set_seed(cfg.seed)
    if getattr(cfg, "num_threads", None):
        torch.set_num_threads(cfg.num_threads)
    device = get_device()
    tag = cfg.tag()
    out_dir = os.path.join(cfg.out_dir, tag)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 62)
    print("设备 : {}".format(device))
    print("模型 : {}".format(cfg.model_name))
    print("权重 : {}".format("预训练权重（对照组）" if cfg.use_pretrained_weights else "随机初始化（从头训练）"))
    print("数据 : {}".format(cfg.dataset_name))
    print("=" * 62)

    tokenizer = datamod.build_tokenizer(cfg)
    train_ds, val_ds, test_ds, names, data_stats, test_texts = datamod.prepare(cfg, tokenizer)
    if not names or len(names) != cfg.num_labels:
        names = ["class_{}".format(i) for i in range(cfg.num_labels)]
    collator = datamod.make_collator(tokenizer)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              collate_fn=collator, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size * 2, shuffle=False,
                            collate_fn=collator, num_workers=cfg.num_workers)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size * 2, shuffle=False,
                             collate_fn=collator, num_workers=cfg.num_workers)

    model, mode = modelmod.build_model(cfg)
    model.to(device)

    total_p, trainable_p = modelmod.count_params(model)
    arch = modelmod.arch_summary(model)
    pbreak = modelmod.param_breakdown(model)
    init_check = modelmod.verify_random_init(model, cfg.num_labels)

    print("架构 : {} / hidden={} / layers={} / heads={}".format(
        arch["arch"], arch["hidden_size"], arch["num_layers"], arch["num_heads"]))
    print("参数 : {:.2f}M (可训练 {:.2f}M)".format(total_p / 1e6, trainable_p / 1e6))
    print("初始化自检 : embedding 首行前 5 个值 = {}".format(init_check["embedding_first_row_head"]))
    print("             理论初始 loss = ln({}) = {}".format(cfg.num_labels, init_check["theoretical_init_loss"]))
    print("样本 : train={} val={} test={}".format(
        len(train_ds), len(val_ds), len(test_ds)))
    print("-" * 62)

    # 优化器与调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)
    total_steps = max(1, len(train_loader) * cfg.epochs)
    warmup_steps = int(total_steps * cfg.warmup_ratio)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda s: _lr_factor(s, warmup_steps, total_steps)
    )

    step_losses, history = [], []
    best_val_loss = float("inf")
    best_path = os.path.join(out_dir, "best_model.pt")
    t0 = time.time()
    global_step = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running, seen = 0.0, 0
        epoch_t = time.time()

        for batch in train_loader:
            inputs, labels = split_batch(batch, device)
            out = model(**inputs, labels=labels)
            loss = out.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            step_losses.append(round(float(loss), 4))
            running += float(loss) * labels.size(0)
            seen += labels.size(0)
            global_step += 1

            if global_step % cfg.log_every == 0:
                lr_now = scheduler.get_last_lr()[0]
                print("  epoch {} step {:>5} loss {:.4f} lr {:.2e}".format(
                    epoch, global_step, float(loss), lr_now))

        train_loss = running / max(seen, 1)
        val = evaluate(model, val_loader, device)
        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val["loss"], 4),
            "val_acc": round(val["acc"], 4),
            "val_macro_f1": round(val["macro_f1"], 4),
            "lr": scheduler.get_last_lr()[0],
            "epoch_sec": round(time.time() - epoch_t, 1),
        })
        print("epoch {}/{}  train_loss {:.4f}  val_loss {:.4f}  val_acc {:.4f}  ({:.1f}s)".format(
            epoch, cfg.epochs, train_loss, val["loss"], val["acc"], time.time() - epoch_t))

        if val["loss"] < best_val_loss:
            best_val_loss = val["loss"]
            torch.save(model.state_dict(), best_path)
            print("  -> 保存最佳模型 (val_loss {:.4f})".format(best_val_loss))

    duration = time.time() - t0

    # 用最佳模型测测试集
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
    test = evaluate(model, test_loader, device)

    cm = confusion_matrix(test["labels"], test["preds"],
                          labels=list(range(cfg.num_labels)))
    rep = classification_report(test["labels"], test["preds"],
                                labels=list(range(cfg.num_labels)),
                                target_names=names, digits=4, zero_division=0)

    print("-" * 62)
    print("测试集结果")
    print(rep)
    print("Accuracy {:.4f}   Macro-F1 {:.4f}".format(test["acc"], test["macro_f1"]))
    print("随机基线（均匀猜测）= {:.4f}".format(1.0 / cfg.num_labels))
    print("总耗时 {:.1f} 秒".format(duration))

    # 出图
    plot_step_loss(step_losses, os.path.join(out_dir, "fig_step_loss.png"))
    plot_epoch_curve(history, os.path.join(out_dir, "fig_epoch_curve.png"))
    plot_confusion(cm, names, os.path.join(out_dir, "fig_confusion.png"))
    n_err = dump_errors(test_texts, test["labels"], test["preds"],
                        test["probs"], names, os.path.join(out_dir, "errors.csv"))

    metrics = {
        "config": cfg.to_dict(),
        "mode": mode,
        "device": str(device),
        "torch_version": torch.__version__,
        "arch": arch,
        "params": {"total": total_p, "trainable": trainable_p,
                   "breakdown": pbreak["detail"]},
        "init_check": init_check,
        "data": dict(data_stats, class_distribution=datamod.class_distribution(train_ds)),
        "train": {"total_steps": total_steps, "warmup_steps": warmup_steps,
                  "duration_sec": round(duration, 1), "history": history},
        "test": {
            "accuracy": round(float(test["acc"]), 4),
            "macro_f1": round(float(test["macro_f1"]), 4),
            "loss": round(float(test["loss"]), 4),
            "random_baseline": round(1.0 / cfg.num_labels, 4),
            "confusion_matrix": cm.tolist(),
            "label_names": names,
            "report": rep,
            "num_errors_dumped": n_err,
        },
        "step_loss_count": len(step_losses),
    }
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("-" * 62)
    print("产物已写入 {}/".format(out_dir))
    for fn in sorted(os.listdir(out_dir)):
        print("   ", fn)
    return metrics


def _lr_factor(step, warmup, total):
    """线性 warmup + 线性衰减。就是 get_linear_schedule_with_warmup 做的事，
    自己写一遍便于在报告里说清楚。"""
    if warmup > 0 and step < warmup:
        return step / max(1, warmup)
    return max(0.0, (total - step) / max(1, total - warmup))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", default="default", choices=["quick", "default", "full"])
    p.add_argument("--pretrained", action="store_true",
                   help="加载官方预训练权重，作为对照实验")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--max-length", type=int, default=None, dest="max_length")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    cfg = {"quick": cfgmod.preset_quick,
           "default": cfgmod.preset_default,
           "full": cfgmod.preset_full}[a.preset]()

    if a.pretrained:
        cfg.use_pretrained_weights = True
        # 微调的学习率要比从零训练小一个量级，否则会把预训练特征冲掉
        cfg.lr = 5e-5
    if a.epochs is not None:
        cfg.epochs = a.epochs
    if a.lr is not None:
        cfg.lr = a.lr
    if a.batch_size is not None:
        cfg.batch_size = a.batch_size
    if a.max_length is not None:
        cfg.max_length = a.max_length
    if a.seed is not None:
        cfg.seed = a.seed
    if a.model is not None:
        cfg.model_name = a.model
        # 传的是本地目录就改加载路径；传的是仓库名则保留 local_model_dir 默认值，
        # 让它继续走"本地有就用本地"的回落逻辑。
        if os.path.isdir(a.model):
            cfg.local_model_dir = a.model
    if a.out is not None:
        cfg.out_dir = a.out

    run(cfg)


if __name__ == "__main__":
    main()
