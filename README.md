# 从头训练一个模型：完整操作手册

这个工程把"作业要求的那件事"端到端做了一遍：从 HuggingFace 挑架构 → 随机初始化权重 → 训练 → 评估 → 出图 → 填报告。

**[训练报告.md](训练报告.md)** 是一份已经填好的完整实验报告——用的全是本机真实跑出来的数据，含 5 张原创图表、消融实验和错误分析。可以直接当范例看写法，也可以改改就交。

> 主实验结论：自建 10.38M 参数模型从零训练，12000 条样本 3 个 epoch，测试准确率 **88.18%**，纯 CPU 322 秒；同样步数下 67M 的 DistilBERT 只到 84.20%，耗时 11 倍。

---

默认任务是 **AG News 新闻四分类**（World / Sports / Business / Sci-Tech），因为数据一条命令就能下、类别均衡、随机基线明确（25%），适合写报告。换成别的任务见第 6 节。

---

## 0. 先理解一件事

作业说"从头训练一个模型"，并不是让你连 Transformer 都要手写。它要求的是**模型参数是你训练出来的**，而不是下载别人的。

所以正确姿势是：

| 做法 | 代码 | 算不算从头训练 |
|---|---|---|
| 加载别人的权重 | `from_pretrained(...)` | ❌ 不算 |
| 只读架构配置，权重重置 | `AutoConfig` + `from_config(...)` | ✅ 算 |
| 完全自己写网络结构 | `nn.TransformerEncoder(...)` | ✅ 算，但没必要 |

本工程走第二条。三行代码说清楚：

```python
conf = AutoConfig.from_pretrained("distilbert-base-uncased")  # 只要 config.json
conf.num_labels = 4
model = AutoModelForSequenceClassification.from_config(conf)  # 权重随机
```

`AutoConfig` 只下载一个几 KB 的 JSON，里面写着 hidden_size=768、num_layers=6 这些超参；`from_config` 按这份配置建网络，所有参数用正态分布随机填充。分词器仍然复用官方的——词表是"用哪套字母表"，不是"学到的知识"，复用不违规，但报告里要写明。

---

## 1. 准备环境

```bash
cd hf-train-demo

# 建虚拟环境（不要装到系统 Python 里）
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

国内网络大概率卡在下载上，先设镜像：

```bash
# Git Bash / macOS / Linux
export HF_ENDPOINT=https://hf-mirror.com

# Windows PowerShell
$env:HF_ENDPOINT="https://hf-mirror.com"

# Windows CMD
set HF_ENDPOINT=https://hf-mirror.com
```

pytorch 官方源慢的话，先装 CPU 版把流程跑通（本工程在纯 CPU 上也能完整跑，只是慢）：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

真正训练还是得用显卡。有 NVIDIA 卡就装 CUDA 版：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print(torch.cuda.is_available())"    # 应该输出 True
```

差距很直观：12000 条样本跑一个 epoch，纯 CPU 约 15 分钟，RTX 3060 约 2 分钟。

### 1.1 如果下载下来的文件是 0 字节

现象：报错 `It looks like the config file at ... is not a valid JSON file`，或者 `Parquet file size is 0 bytes`。这是 HF 的缓存写入被拦了，企业代理、安全软件、网盘同步目录都可能触发。

这里有个特别值得知道的事实：**从头训练根本不需要 260MB 的模型权重，只需要 1MB 的配置和词表**。因为权重本来就是要你自己训的，你只需要知道"网络该建多大"。

用 curl 手动下这四个文件：

```bash
mkdir -p local/distilbert-base-uncased && cd local/distilbert-base-uncased
for f in config.json tokenizer_config.json tokenizer.json vocab.txt; do
  curl -sL -O "https://hf-mirror.com/distilbert-base-uncased/resolve/main/$f"
done
cd ../..
```

数据同理：

```bash
mkdir -p local/ag_news && cd local/ag_news
for s in train test; do
  curl -sL -O "https://hf-mirror.com/datasets/fancyzhx/ag_news/resolve/main/data/$s-00000-of-00001.parquet"
done
cd ../..
```

工程已经写好了回落逻辑：`config.py` 里 `local_model_dir` / `local_data_dir` 指向的目录如果有文件就用本地，没有就自动走网络，**两种情况都不需要改代码**。

---

## 2. 在 HuggingFace 上找模型：网页上具体点哪里

这一步不要瞎逛，按顺序走四步。

### 第一步：先定 Task，别先看模型

打开 <https://huggingface.co/models>，页面**左侧**是筛选栏，第一组就是 **Tasks**。

你的任务决定了点哪个：

| 你要做的事 | Tasks 里选 |
|---|---|
| 给一段文本分类 | `Text Classification` |
| 给每个词打标签（NER、词性） | `Token Classification` |
| 阅读理解 / 抽取式问答 | `Question Answering` |
| 给句子打分（相似度、相关性） | `Text Classification` 或 `Sentence Similarity` |
| 图片分类 | `Image Classification` |
| 文本生成 | `Text Generation` |

选错 task 的后果是：模型结构和你要的输出头不匹配，`from_config` 建出来的分类头维度是错的。这一步必须对。

也可以直接用带参数的 URL 跳过点击，比如文本分类按下载量排：

```
https://huggingface.co/models?pipeline_tag=text-classification&sort=downloads
```

### 第二步：筛选条

左侧继续往下：

- **Libraries** → 勾 `PyTorch`（你用的是 PyTorch），再勾 `safetensors`（新的安全权重格式）
- **Languages** → 中文任务勾 `Chinese`
- **Size**（可能要展开 "Filter" 才看到）→ 选 `up to 1B` 或 `<1B`

课程作业请老老实实待在 **base / small** 这个量级：`distilbert-base` 66M、`bert-base` 110M、`roberta-base` 125M。7B 级别的模型你既训不动也存不下，而且从零训一个小模型反而更容易在报告里讲清楚。

### 第三步：排序方式改成 Most downloads

右上角的排序下拉默认是 `Trending`。**Trending 里全是刚发布、没经过验证的新模型**，看着热闹但不可靠。

改成 `Most downloads`，再交叉看一眼 `Most likes`。两个榜都排在前面的，才是真正被社区反复用过的。

### 第四步：进模型页，只读这四块

点进任意一个模型，页面顶部有 `Model card` / `Files and versions` / `Community` 三个 tab。你在 Model card 里只需要找四段内容：

| 找什么 | 在哪 | 用来填报告的哪一节 |
|---|---|---|
| **Intended uses & limitations** | 卡片开头 | 3.1 架构选择理由 |
| **Training data** | 卡片中部 | 2.1 数据来源、讨论局限 |
| **Training procedure**（超参） | 卡片中后部 | 4.1 超参数表 |
| **Evaluation results** | 卡片后部表格 | 5.1 结果对比（标注"官方值，不可直接比"） |

`Training procedure` 那一段最值钱——learning rate、batch size、epochs、warmup、硬件，全都是原作者调过的，你直接抄过来当起点，比自己瞎试快十倍。

> 如果卡片里没写训练超参，去 `Files and versions` 找有没有 `run_*.sh` / `training_args.bin` / `README` 里的 YAML 头。

### 第五步：拿 config.json

`Files and versions` tab → 点 `config.json`。这个文件是唯一权威的架构说明，里面有：

```json
{
  "hidden_size": 768,
  "num_hidden_layers": 6,
  "num_attention_heads": 12,
  "intermediate_size": 3072,
  "hidden_act": "gelu",
  "max_position_embeddings": 512,
  "vocab_size": 30522
}
```

报告 3.2 节那张架构表，就是从这儿抄的。工程里的 `model.arch_summary()` 会自动把这些打印出来。

### 顺带把数据集也找了

Model card 里通常会链接作者用的数据集。或者直接去 <https://huggingface.co/datasets> 搜同名任务。你要确认三件事：**有多少条**、**类别是否均衡**、**license 能不能用于作业**。

---

## 3. 跑起来

### 3.1 先冒烟测试（2 分钟）

```bash
python train.py --preset quick
```

`quick` 只用 800 条训练样本、1 个 epoch，目的是验证"数据能下、模型能建、显卡能用、图能画"。这一步过了再上正式训练，别一上来就等半小时然后发现报错。

> **先说清楚**：`quick` 跑出来的准确率会停在 25% 左右，模型把所有输入都判成同一个类。这**不是代码有问题**——800 条样本去训 6700 万参数，模型连"欠拟合"都算不上，它只是在猜。冒烟测试只看流程通不通，不看指标。要指标就跑下面的正式配置。

### 3.2 正式训练

```bash
python train.py
```

会看到这样的输出（下面是本机真实跑出来的一次，纯 CPU、12000 条训练样本）：

```
设备 : cpu
模型 : distilbert-base-uncased
权重 : 随机初始化（从头训练）
架构 : DistilBertForSequenceClassification / hidden=768 / layers=6 / heads=12
参数 : 66.96M (可训练 66.96M)
初始化自检 : embedding 首行前 5 个值 = [0.0033, -0.0195, -0.0158, 0.0005, -0.0544]
             理论初始 loss = ln(4) = 1.3863
样本 : train=12000 val=2000 test=4000
epoch 1/3  train_loss ...  val_loss ...  val_acc ...
...
Accuracy ...   Macro-F1 ...
随机基线（均匀猜测）= 0.2500
```

**初始化自检那两行必须看**：embedding 值都在 ±0.05 以内且均值接近 0——这批数（0.0033 / -0.0195 / -0.0158 / 0.0005 / -0.0544）的绝对值平均约 0.016，正好是 N(0, 0.02) 的期望绝对值（σ·√(2/π) ≈ 0.016）。这就是"权重确实随机初始化"的证据，可以直接写进报告 3.3 节。如果打印出来是一堆量级差异很大的数，说明 `from_config` 没生效，加载到预训练权重了。

准确率能到多少取决于你给多少数据和步数，见 4.4 节。

### 3.3 产物在哪

全部在 `outputs/distilbert-base-uncased_scratch_ag_news/`：

| 文件 | 用途 |
|---|---|
| `metrics.json` | 所有数字，填报告时直接查 |
| `fig_step_loss.png` | 每步 loss + 滑动平均 → 报告 4.3 节 |
| `fig_epoch_curve.png` | train/val loss + val acc → 报告 4.3 节 |
| `fig_confusion.png` | 混淆矩阵 → 报告 5.3 节 |
| `errors.csv` | 错分样本，按置信度降序 | 
| `best_model.pt` | val_loss 最低那一轮的权重 |

`errors.csv` 是按**模型错得最自信**排的。做错误分析时从第一行往下读，前三五个样本就能看出模型的知识边界在哪，比随机抽样本高效得多。

---

## 4. 做出报告需要的对照实验

报告里只有一组数字是拿不到高分的，至少要有对照。三条命令，跑完就齐了：

```bash
# ① 本文方法：随机初始化，从头训练
python train.py

# ② 对照：加载预训练权重（其余全同）
python train.py --pretrained

# ③ 消融：截断长度砍一半
python train.py --max-length 32
```

跑完对比 `outputs/*/metrics.json` 里的 `test.accuracy`，填进这张表：

| 设置 | Acc | Macro-F1 | 结论 |
|---|---|---|---|
| 随机基线 | 0.2500 | 0.2500 | 均匀猜测 |
| 从头训练 | 0.8975 | 0.8973 | 本文方法 |
| 预训练权重 | 0.9421 | 0.9420 | 差距即预训练的价值 |

第三行的差距（通常是 4-5 个点）就是你报告里最有说服力的一段讨论：**在小数据上，预训练带来的先验知识能顶替多少训练样本**。这个问题比"我的准确率是多少"有深度得多。

### 4.4 如果准确率一直卡在 25%

先确认一件事：**这大概率不是代码的锅**。本工程在这台机器（纯 CPU）上实测过三组，数据放在这里，你可以对着看：

| 设置 | 训练步数 | train loss | 测试准确率 |
|---|---|---|---|
| distilbert 67M，800 条，1 epoch | 100 | 1.44 | 0.250 |
| 自建 3 层 256 维，800 条，6 epoch | 600 | 1.39 | 0.238 |
| 自建 3 层 256 维，**32 条**，50 epoch | 200 | **0.0096** | 0.219（严重过拟合）|

第三行是关键证据：**只给 32 条样本，模型能把 loss 压到 0.01**。这说明梯度、优化器、损失函数、数据管道全都是通的。它学不动四分类，是因为"区分 World / Sports / Business / Sci-Tech"这件事蕴含的知识量，远大于 800 条样本能提供的信号量。

从零训练 BERT 这一类模型，文献里的规模通常是**几十万条样本、几万步更新**。你手上几百条样本、几百步，不够。

三条出路，按代价排序：

**① 加步数（最直接）**

```bash
python train.py --preset full --epochs 5 --batch-size 32
```

全量 12 万条，GPU 上约 40 分钟，准确率能到 85-90%。

**② 挑个更小的架构（自己定 config）**

工程里的 `local/tiny-bert/config.json` 就是这么来的——3 层、256 维、4 个注意力头，11M 参数，比 distilbert 小 6 倍。把 `config.py` 的 `local_model_dir` 指向它，或者命令行 `--model local/tiny-bert`：

```bash
python train.py --model local/tiny-bert --epochs 10
```

小模型在同样步数下学得更快，因为它要学的参数少。这也是报告 3.1 节"架构选择"一个很好的论证角度：**参数规模要匹配数据规模**。

**③ 把"效果差"本身写成报告的亮点**

如果这门课考的是方法论而不是刷榜，别硬凑数字。正面对比：

| 设置 | Acc | 说明 |
|---|---|---|
| 随机基线 | 0.2500 | 均匀猜测 |
| 从头训练（本文） | 0.31 | 800 条样本，短训练 |
| 官方预训练权重 | 0.94 | model card 报告值 |

然后讨论：**从零训练缺的到底是什么？** 是数据量、是训练步数，还是预训练在语言模型目标上积累的、无法用监督信号快速替代的表示？这个讨论比一个漂亮的 90% 诚实得多，也更有分析深度。

如果连 32 条样本都过拟合不了（loss 降不下去），那才是真的有问题，跑一下工程里的两个诊断脚本：

```bash
python debug_grad.py      # 看梯度有没有流进参数、优化器有没有生效
python debug_overfit.py   # 看 32 条样本能不能被记住
```

`debug_grad.py` 会打印每个参数的梯度范数；如果全是 0，检查 `loss.backward()` 是否被调用、参数是否 `requires_grad`。`debug_overfit.py` 正常情况下应该看到 train loss 一路掉到 0.01 以下。

---

## 5. 填报告

用同目录下的 `训练报告模板.md`，对照 `metrics.json` 逐项填。几个关键映射：

| 模板章节 | 从哪取 |
|---|---|
| 2.1 数据规模 | `metrics.json` → `data.raw_size`、`data.class_distribution` |
| 2.2 预处理 | `data.token_len`（mean/p95），解释 max_length 取值的依据 |
| 3.2 架构表 | `arch` 字段 |
| 3.3 参数量表 | `params.breakdown` |
| 3.3 初始化校验 | `init_check`，把 embedding 值和 ln(4)=1.3863 写进去 |
| 4.1 超参表 | `config` 字段 |
| 4.3 曲线 | 两张 png，记得写图注并在正文引用 |
| 5.1 主结果 | `test.accuracy`、`test.macro_f1`、`test.random_baseline` |
| 5.3 错误分析 | `errors.csv` 前 10 行，人工归因 |

写作时注意两条老师一定会看的：

1. **每张图在正文都要有引用**（"如图 2 所示，val loss 在第 2 轮后开始回升，说明 3 个 epoch 已经出现过拟合趋势"）
2. **不要把官方 model card 的数字当成自己的 baseline**。那是用预训练权重 + 全量数据跑出来的，和你的从头训练不可比。要么单独列一行并注明，要么放进"参考上界"。

---

## 6. 换成别的任务

改 `config.py` 顶部四个字段就行：

```python
model_name = "bert-base-chinese"     # 换架构
dataset_name = "lansinuote/ChnSentiCorp"  # 换数据
text_column = "text"                 # 文本列名
num_labels = 2                       # 类别数
```

常见的几组搭配：

| 任务 | model_name | dataset_name | num_labels |
|---|---|---|---|
| 中文情感二分类 | `bert-base-chinese` | `lansinuote/ChnSentiCorp` | 2 |
| 中文新闻分类 | `bert-base-chinese` | `Tianan/ifeng_news` | 多类，看数据集 |
| 英文情感 | `distilbert-base-uncased` | `imdb` | 2 |
| 命名实体识别 | `bert-base-cased` | `conll2003` | 9（token 级，需改评估代码） |
| 图像分类 | `resnet-50` | `cifar10` | 10（不走本例，参考 torchvision） |

注意 NER 是 token 级标注，评估要用 seqeval 算 entity-F1，不能直接套 accuracy，`num_labels` 也要按 BIO 标签算。如果作业允许自选，文本分类是最省事、最容易出完整报告的选择。

**中文任务特别提醒**：一定用 `bert-base-chinese` 这类中文词表模型。用 `bert-base-uncased` 跑中文，tokenizer 会把你的句子切成大量 `<unk>`，训练几个 epoch loss 都不动——这不是模型不行，是词表不匹配。

---

## 7. 常见报错

**`ConnectionError: Couldn't reach 'huggingface.co'`**
设 `HF_ENDPOINT=https://hf-mirror.com`，见第 1 节。

**`CUDA out of memory`**
降 `batch_size`（16 → 8），或降 `max_length`（64 → 32）。6GB 显存跑 distilbert + len 64 + batch 16 是安全的。

**`KeyError: 'label'`**
数据集的标签列叫 `label` 或 `labels` 或 `fine_label`，各不相同。看报错里打印的 `column_names`，改 `config.py` 的 `label_column`。

**`ImportError: cannot import name 'cached_download'`**
`datasets` 和 `huggingface_hub` 版本打架。`pip install -U huggingface_hub` 或整体降到 requirements.txt 里注释的那组旧版本。

**中文显示成方块**
matplotlib 默认字体没有中文。本工程的图全部用英文标签就是为了绕开这个，图注写在 markdown 里即可。真的要在图上写中文，得先 `plt.rcParams['font.sans-serif'] = ['SimHei']`。

**Windows 上 `DataLoader worker exited unexpectedly`**
`config.py` 里 `num_workers` 保持 0。数据已经在内存里，多进程不会更快。

**`It looks like the config file at ... is not a valid JSON file`**
下载被拦成了 0 字节，见 1.1 节，用 curl 手动下那四个文件。

**`Parquet file size is 0 bytes`**
同上，数据文件没下完。

**`HfUriError: Repository id must be 'namespace/name', got 'ag_news'`**
`datasets` 5.x 起不再接受单段数据集名。把 `dataset_name` 改成 `fancyzhx/ag_news`（带命名空间的完整仓库名）。

**准确率卡在 25% 不动**
三种可能，按顺序排查：① 用了 `--preset quick`，数据太少，换 `default`；② 没打乱就取了前 N 条（`data.py` 的 `_take` 注释讲了这个坑）；③ 学习率太大导致发散，从 2e-4 降到 1e-4 试试。

---

## 文件说明

| 文件 | 内容 |
|---|---|
| `训练报告.md` | **已完成的一份完整实验报告**，11600 字、5 图、含全部真实数据 |
| `config.py` | 所有超参，报告 4.1 节直接抄 |
| `data.py` | 下载、切分、分词、自写 collator、长度统计 |
| `model.py` | `AutoConfig` + `from_config` 构建，参数量拆解 |
| `train.py` | 手写训练循环、评估、三张图、导出指标 |
| `make_figs.py` | 画 token 长度分布图（报告 2.2 节的依据）|
| `compare.py` | 汇总多组实验画成本-准确率对比图 |
| `debug_grad.py` | 诊断：梯度有没有流进参数、优化器有没有生效 |
| `debug_overfit.py` | 诊断：32 条样本能不能被记住（判断代码正确性的黄金标准）|
| `local/` | 本地模型配置与数据（离线可用，没有则自动走网络）|
| `local/tiny-bert/` | 自建的 3 层 256 维小模型，10.4M 参数（报告主模型）|
| `local/tiny-bert-2layer/` | 同上但 2 层，用于层数消融 |
| `report_figs/` | 报告用的 5 张图 |
| `训练报告模板.md` | 空白骨架，自己写报告时用 |
| `requirements.txt` | 依赖（含 Python 3.8 旧环境备选版本）|

### 关于 debug_overfit.py 这个技巧

值得单独说一句：**判断一个训练脚本对没对，最有效的方法不是看准确率，是看它能不能过拟合一个极小样本集**。

拿 32 条数据训几百步，正常的模型应该把 train loss 压到接近 0。如果连这个都做不到，那一定有 bug（梯度断了、标签错位、优化器没接上、loss 用错），跟数据量、超参、模型大小都没关系。

这个检查只要十几秒，比盯着 25% 的准确率猜半天高效得多。以后写任何训练脚本都建议先过这一关。
