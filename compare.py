"""把所有实验汇总成一张图。

横轴是训练耗时，纵轴是测试准确率，点的大小代表参数量。
这张图能一眼看出帕累托前沿在哪：哪些配置是"又快又准"的。
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = [
    ("DistilBERT 6L/768\n67.0M · 1 ep",
     "outputs_ctrl/distilbert-base-uncased_scratch_ag_news/metrics.json", "#D85A30", (0, 28), "center"),
    ("3L/256 · 10.4M · 1 ep",
     "outputs_probe/local_tiny-bert_scratch_ag_news/metrics.json", "#378ADD", (30, 16), "left"),
    ("3L/256 · 10.4M · 3 ep",
     "outputs_report/local_tiny-bert_scratch_ag_news/metrics.json", "#185FA5", (0, -30), "center"),
    ("2L/256 · 9.6M · 3 ep",
     "outputs_abl_layer/local_tiny-bert-2layer_scratch_ag_news/metrics.json", "#0F6E56", (0, 24), "center"),
    ("3L/256 · len=32 · 3 ep",
     "outputs_abl_len/local_tiny-bert_scratch_ag_news/metrics.json", "#888780", (0, 24), "center"),
]


def main():
    pts = []
    for label, path, color, off, ha in RUNS:
        if not os.path.exists(path):
            print("跳过（文件不存在）:", path)
            continue
        m = json.load(open(path, encoding="utf-8"))
        pts.append({
            "label": label,
            "sec": m["train"]["duration_sec"],
            "acc": m["test"]["accuracy"],
            "params": m["params"]["total"] / 1e6,
            "color": color,
            "off": off,
            "ha": ha,
        })

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for p in pts:
        ax.scatter(p["sec"], p["acc"], s=p["params"] * 13, color=p["color"],
                   alpha=0.8, edgecolor="white", linewidth=1.2, zorder=3)
        ax.annotate(p["label"], (p["sec"], p["acc"]), textcoords="offset points",
                    xytext=p["off"], ha=p["ha"], fontsize=8.2, color="#2C2C2A",
                    linespacing=1.5, zorder=4)

    ax.set_xscale("log")
    ax.set_xlabel("Training time (seconds, log scale)")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Test accuracy vs training cost (bubble size = parameter count)", pad=12)
    ax.set_ylim(0.825, 0.905)
    ax.set_xlim(80, 1600)
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    out = "report_figs/fig4_cost_accuracy.png"
    os.makedirs("report_figs", exist_ok=True)
    fig.savefig(out, dpi=200)
    print("已保存:", out)
    for p in sorted(pts, key=lambda x: -x["acc"]):
        print("  {:>26}  acc={:.4f}  {:>6.0f}s  {:>5.1f}M".format(
            p["label"].replace("\n", " "), p["acc"], p["sec"], p["params"]))


if __name__ == "__main__":
    main()
