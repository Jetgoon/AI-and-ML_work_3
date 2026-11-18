"""全局配置：所有超参数集中在这里。

写报告时的"超参数表"直接抄这个文件，保证代码和文档一致。
修改实验设置只需要改这个文件，不用动 train.py。
"""

from dataclasses import dataclass, asdict
import os
from typing import Optional


@dataclass
class Config:
    # ========== 模型 ==========
    # 从 HuggingFace 找架构名，例如 distilbert-base-uncased / bert-base-chinese
    model_name: str = "distilbert-base-uncased"

    # ★★ 核心开关 ★★
    # False = 只取架构配置，权重随机初始化（这才是"从头训练"）
    # True  = 加载官方预训练权重（用来做对照实验，证明预训练的价值）
    use_pretrained_weights: bool = False

    # ========== 数据 ==========
    dataset_name: str = "ag_news"
    text_column: str = "text"
    label_column: str = "label"
    num_labels: int = 4
    max_length: int = 64

    # 数据规模（None 表示用全量）。默认值保证在无 GPU 的机器上也能跑完。
    train_size: Optional[int] = 12000
    val_size: Optional[int] = 2000
    test_size: Optional[int] = 4000

    # ========== 训练 ==========
    batch_size: int = 16
    # 从零训练用 2e-4 起步；如果加载预训练权重微调，要降到 5e-5。
    # 用 2e-4 去微调预训练模型会把它辛苦学到的特征冲掉，loss 反而上升。
    lr: float = 2e-4
    epochs: int = 3
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    seed: int = 42

    # ========== 运行 ==========
    out_dir: str = "outputs"
    log_every: int = 50          # 每多少个 step 打印一次
    num_workers: int = 0         # Windows 上保持 0，最稳
    # CPU 训练时的 torch 线程数。小模型线程开太多，线程同步开销会盖过计算收益，
    # 8 通常比默认的 14 快。用 GPU 时这个设置无影响。
    num_threads: int = 8

    # ========== 离线兜底 ==========
    # 如果这两个目录存在且文件完整，就直接读本地，不走网络。
    # 网络正常时它们会被忽略，行为和不设置完全一样。
    local_model_dir: Optional[str] = "local/distilbert-base-uncased"
    local_data_dir: Optional[str] = "local/ag_news"

    def model_source(self):
        """返回实际用于加载的路径：本地目录优先，否则回落到仓库名。"""
        if self.local_model_dir and os.path.exists(
                os.path.join(self.local_model_dir, "config.json")):
            return self.local_model_dir
        return self.model_name

    def data_source(self):
        """返回 (数据集名, 本地文件字典)。本地文件字典为空时走网络。"""
        if self.local_data_dir and os.path.isdir(self.local_data_dir):
            files = {}
            for split in ("train", "validation", "test"):
                p = os.path.join(self.local_data_dir, split + ".parquet")
                if os.path.exists(p) and os.path.getsize(p) > 0:
                    files[split] = p
            if files:
                return "parquet", files
        return self.dataset_name, {}

    def to_dict(self):
        return asdict(self)

    def tag(self):
        """用于区分不同实验的输出目录名。"""
        init = "pretrained" if self.use_pretrained_weights else "scratch"
        return "{}_{}_{}".format(self.model_name.replace("/", "_"), init, self.dataset_name)


# ---- 预设：命令行里用 --preset 选择 ----

def preset_quick():
    """冒烟测试用，2 分钟内跑完，只为验证流程通不通。"""
    return Config(train_size=800, val_size=400, test_size=400,
                  max_length=32, epochs=1, batch_size=8, log_every=20)


def preset_default():
    """正式跑，3060 上大约 6-10 分钟。"""
    return Config()


def preset_full():
    """全量数据，报告冲分用。"""
    return Config(train_size=None, val_size=None, test_size=None,
                  max_length=128, epochs=3, batch_size=32)
