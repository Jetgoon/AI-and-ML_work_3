"""诊断脚本：检查梯度是否真的流进了参数。

现象是 train_loss 一直卡在 ln(4)≈1.3863 附近不动。
如果梯度正常，说明是数据/超参问题；如果梯度为 0 或参数不变，说明代码有 bug。
"""

import torch
from torch.utils.data import DataLoader

import config as cfgmod
import data as datamod
import model as modelmod

cfg = cfgmod.preset_quick()
tok = datamod.build_tokenizer(cfg)
tr, va, te, names, stats, texts = datamod.prepare(cfg, tok)
coll = datamod.make_collator(tok)
dl = DataLoader(tr, batch_size=8, shuffle=True, collate_fn=coll)

model, mode = modelmod.build_model(cfg)
model.train()

batch = next(iter(dl))
print("=" * 60)
print("batch keys :", list(batch.keys()))
print("input_ids  :", tuple(batch["input_ids"].shape), batch["input_ids"].dtype)
print("labels     :", batch["labels"].tolist())

inputs = {k: v for k, v in batch.items() if k != "labels"}
labels = batch["labels"]

out = model(**inputs, labels=labels)
print("logits     :", tuple(out.logits.shape))
print("loss       :", round(float(out.loss), 5))
print("logits mean:", round(float(out.logits.mean()), 5),
      "std:", round(float(out.logits.std()), 5))

out.loss.backward()

grads = [(n, p.grad) for n, p in model.named_parameters() if p.grad is not None]
total_gn = sum(float(g.norm()) ** 2 for _, g in grads) ** 0.5
print("有梯度的参数 : {} / {}".format(
    len(grads), sum(1 for _ in model.parameters())))
print("总梯度范数   : {:.6e}".format(total_gn))

zero_like = sum(1 for _, g in grads if float(g.abs().max()) < 1e-12)
print("梯度全为 0 的参数 : {}".format(zero_like))

for n, g in grads[:3]:
    print("   {:<55} |grad|max={:.3e}".format(n, float(g.abs().max())))

print("-" * 60)
opt = torch.optim.AdamW(model.parameters(), lr=5e-4)
before = model.classifier.weight.detach().clone()
opt.step()
delta = float((model.classifier.weight - before).abs().max())
print("一步 AdamW 后 classifier 权重最大变化: {:.6e}".format(delta))
print("（若为 0，说明优化器没作用到参数上）")

print("-" * 60)
model.eval()
with torch.no_grad():
    out2 = model(**inputs)
pred = out2.logits.argmax(-1)
print("更新一步后的预测:", pred.tolist())
print("真实标签        :", labels.tolist())
print("=" * 60)
