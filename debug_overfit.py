"""过拟合测试：判定代码到底有没有 bug。

方法：只给模型 32 条样本，反复训练 200 步。
- 如果 loss 能降到接近 0，说明梯度、优化器、损失函数全都没问题，
  前面准确率上不去纯粹是"数据量不足以让 67M 参数学到东西"。
- 如果 32 条都记不住，那才是代码真的坏了。
"""

import config as cfgmod
import train as trainmod

cfg = cfgmod.preset_quick()
cfg.local_model_dir = "local/tiny-bert"
cfg.model_name = "tiny-bert"
cfg.train_size = 32
cfg.val_size = 32
cfg.test_size = 32
cfg.batch_size = 8
cfg.max_length = 32
cfg.epochs = 50
cfg.lr = 1e-3
cfg.warmup_ratio = 0.05
cfg.log_every = 20
cfg.out_dir = "outputs_dbg"

if __name__ == "__main__":
    trainmod.run(cfg)
