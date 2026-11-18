"""生成报告用的补充图表。

目前只画一张：训练集的 token 长度分布。
这张图是报告 2.2 节"为什么 max_length 取这个值"的唯一有力证据——
没有它，max_length 的取值就只能写"凭经验"，评审一问就露馅。
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset

import config as cfgmod
import data as datamod


def main():
    cfg = cfgmod.preset_default()
    tok = datamod.build_tokenizer(cfg)
    name, files = cfg.data_source()
    raw = load_dataset(name, data_files=files) if files else load_dataset(name)

    n = cfg.train_size or len(raw["train"])
    ds = raw["train"].shuffle(seed=cfg.seed).select(range(min(n, len(raw["train"]))))
    lens = [len(x) for x in tok(list(ds["text"]))["input_ids"]]

    arr = np.array(lens)
    p50 = int(np.percentile(arr, 50))
    p95 = int(np.percentile(arr, 95))
    print("样本数 {}  mean {:.1f}  p50 {}  p95 {}  max {}".format(
        len(arr), arr.mean(), p50, p95, arr.max()))
    print("被 max_length={} 截断的比例: {:.2%}".format(
        cfg.max_length, float((arr > cfg.max_length).mean())))

    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    ax.hist(arr, bins=60, color="#B5D4F4", edgecolor="#185FA5", linewidth=0.4)
    ax.axvline(p50, color="#185FA5", linestyle="--", linewidth=1.4,
               label="median = {}".format(p50))
    ax.axvline(p95, color="#D85A30", linestyle="--", linewidth=1.4,
               label="p95 = {}".format(p95))
    ax.axvline(cfg.max_length, color="#5F5E5A", linestyle="-", linewidth=1.6,
               label="max_length = {}".format(cfg.max_length))
    ax.set_xlabel("Token count per sample")
    ax.set_ylabel("Number of samples")
    ax.set_title("Token length distribution (training set, n={})".format(len(arr)))
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    out = "outputs_report/local_tiny-bert_scratch_ag_news/fig_length_dist.png"
    fig.savefig(out, dpi=200)
    print("已保存:", out)


if __name__ == "__main__":
    main()
