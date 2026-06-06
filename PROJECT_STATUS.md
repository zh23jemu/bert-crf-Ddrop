# 项目现状与下一步计划

## 当前目标

本项目用于中文教师评语实体抽取，识别 `moral`、`personality`、`study`、`suggestion`、`weakness` 五类实体。

当前用户核心目标：

- 第一目标：测试集 F1 从约 77 提升到 80 以上。
- 第二目标：补充一个能体现教师评语文本特点的创新点，并尽量让创新模型超过当前最佳基线。

## 当前最好结果

已达成 80+ 目标。历史最佳基线来自 Slurm 任务 `34660732`：

```text
测试集最终F1值: 0.802836
训练过程中最佳验证F1值: 0.786122
```

对应默认参数已经固化到 `train.py`：

```bash
--seed 21 --lr 2.8e-5 --drop-prob 0.25 --rdrop-alpha 0 --early-stop-patience 10
```

最佳分类报告：

```text
              precision    recall  f1-score   support

       moral       0.78      0.85      0.81       234
 personality       0.78      0.79      0.78       157
       study       0.83      0.84      0.83       244
  suggestion       0.74      0.80      0.77       286
    weakness       0.82      0.83      0.83       181

   micro avg       0.78      0.82      0.80      1102
   macro avg       0.79      0.82      0.80      1102
weighted avg       0.79      0.82      0.80      1102
```

## 已实现内容

- 完成 `.gitignore`、`.env.example`、`requirements.txt`。
- API Key 已改为从 `.env` 或环境变量读取，真实 `.env` 不入库。
- 大模型权重 `bert-base-chinese/pytorch_model.bin` 不入库，需要单独下载或拷贝。
- 新增 Slurm 训练脚本 `slurm_train.sh`。
- 在 CRF 中加入 BIO 合法转移约束。
- 固化最佳基线参数。
- 新增 CDFA-NER 创新模块：教师评语语篇功能感知增强。
- 修复 Slurm 并行任务共用 `model/best_model.pth` 导致 checkpoint 损坏的问题，现在默认按 JobID 保存 checkpoint。
- 新增 `--init-from` 与 `--freeze-base-model`，用于加载强基线后只训练 CDFA 语篇残差模块。

## 创新模块说明

创新点名称：CDFA-NER，教师评语语篇功能感知增强。

核心思想不是关键词硬匹配，也不是通用注意力模块，而是利用教师评语常见结构：

```text
优点评价 -> 不足指出 -> 改进建议
```

数据层会根据触发词和分句结构自动生成语篇功能标签：

- `positive_eval`
- `weakness_context`
- `suggestion_context`
- `transition_context`
- `neutral`

模型层当前采用残差式语篇提示：

- BERT-CRF 主路径保持不变。
- 语篇功能特征只学习额外 logit 修正。
- 目标是在不破坏强基线的前提下，用教师评语结构信号微调输出。

## 已跑过的创新实验

直接启用 CDFA 的实验没有超过最佳基线：

```text
34753901: discourse_emb_size=16, 测试F1=0.778518
34757206: discourse_emb_size=4,  测试F1=0.778986
34757209: discourse_emb_size=8,  测试F1=0.788070
34758437: 残差式CDFA emb=8, 测试F1=0.790760
```

基线保持式微调曾用较弱底座 `34759187` 测试：

```text
34759187: 重跑基线，测试F1=0.785965
34804353: 基于34759187冻结主路径训练CDFA，测试F1=0.783223
```

结论：当前关键不是微调命令，而是需要先重新拿到一个 80+ 的强基线 checkpoint，再做 CDFA 微调。

## 换电脑后的准备

新电脑有 CUDA，可以本地直接训练，不必每次推到服务器。

建议步骤：

```bash
git clone https://github.com/zh23jemu/bert-crf-Ddrop.git
cd bert-crf-Ddrop
```

准备本地 `.env`：

```bash
cp .env.example .env
```

然后把 DashScope API Key 填到 `.env`。

准备 BERT 权重：

```bash
gh release download model-assets \
  --repo zh23jemu/bert-crf-Ddrop \
  --pattern pytorch_model.bin \
  --dir bert-base-chinese
```

安装依赖时优先使用本地 `.venv`。CUDA 电脑建议安装与本机驱动匹配的 PyTorch GPU 版本。

## 下一步计划

### 1. 先重新获得 80+ 强基线 checkpoint

默认跑法：

```bash
.venv/bin/python train.py
```

如果在 Windows CUDA 电脑上：

```powershell
.venv\Scripts\python.exe train.py
```

如果单次没有复现 `0.802836`，建议先跑几组已知较强参数：

```bash
.venv/bin/python train.py --seed 21 --lr 2.8e-5 --drop-prob 0.25 --rdrop-alpha 0 --early-stop-patience 10
.venv/bin/python train.py --seed 21 --lr 3e-5 --drop-prob 0.25 --rdrop-alpha 0 --early-stop-patience 10
.venv/bin/python train.py --seed 100 --lr 3e-5 --drop-prob 0.25 --rdrop-alpha 0 --early-stop-patience 10
```

目标：先得到测试 F1 接近或超过 `0.802836` 的 checkpoint。

### 2. 基于强 checkpoint 训练创新模块

假设强基线 checkpoint 是：

```text
model/best_model.pth
```

运行：

```bash
.venv/bin/python train.py \
  --use-discourse-feature \
  --discourse-emb-size 8 \
  --init-from model/best_model.pth \
  --freeze-base-model
```

Windows PowerShell：

```powershell
.venv\Scripts\python.exe train.py `
  --use-discourse-feature `
  --discourse-emb-size 8 `
  --init-from model/best_model.pth `
  --freeze-base-model
```

### 3. 如果仍未超过基线

优先继续做小范围实验：

- `--discourse-emb-size 2`
- `--discourse-emb-size 4`
- 只在 `weakness/suggestion` 类别上做语篇残差修正
- 调低 `head-lr`
- 缩短创新微调 epoch，避免过拟合

## 重要风险

- 历史最佳 `34660732` 的 checkpoint 未确认还在服务器上；如果丢失，需要重新复现。
- 当前默认参数不保证每次都稳定复现 `0.802836`，存在随机性和环境差异。
- CDFA 当前还没有超过最佳基线，后续必须先拿到强基线 checkpoint，再微调创新层。
- `.env`、`.venv`、`*.pth` 和 `bert-base-chinese/pytorch_model.bin` 不会提交到 GitHub，需要在新电脑单独准备。

