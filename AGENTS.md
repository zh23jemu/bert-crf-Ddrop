# 项目维护记录

## 项目目标

本项目用于中文教师评语中的实体抽取，识别性格特质、品德行为、学习表现、不足之处和期望建议等实体类型。

## 技术栈

- Python
- PyTorch
- Transformers
- TorchCRF
- OpenAI SDK 兼容接口（DashScope）
- Pandas、Jieba、Matplotlib、Seqeval、Scikit-learn

## 当前架构

- `main.py`：调用大模型接口辅助生成或补充标注数据。
- `preprocess.py`：将 JSONL 标注数据转换为 BIO 序列。
- `dataset.py`：读取 BIO 数据、执行 BERT tokenizer 对齐和 batch padding。
- `model/model.py`：定义 BERT + CRF 序列标注模型。
- `train.py`：负责训练、评估、R-Drop 正则和最佳模型保存。
- `clean.py`、`count.py`、`text.py`、`visual.py`：负责评语清洗、数据统计和标签分布可视化。

## 开发规范

- 默认使用项目本地 `.venv`，不要直接使用系统 Python。
- API Key 等敏感配置写入本地 `.env`，不要硬编码到源码或提交到 Git。
- `.gitignore` 需要持续维护，避免提交缓存、IDE 配置、虚拟环境、大模型权重和训练检查点。
- 修改代码前先读取相关文件，保持最小修改，不做无关重构。

## Current Status

已完成初版 Git 提交、仓库本地配置整理，并开始围绕验证集 F1 从约 77 提升到 80+ 的目标优化训练流程。

## Recent Changes

- 新增 `.gitignore`，忽略 Python 缓存、IDE 配置、本地虚拟环境、环境变量文件和训练权重/检查点。
- 将 `main.py` 中的 DashScope API Key 改为从 `.env` 或系统环境变量读取。
- 新增 `.env.example` 作为本地配置模板。
- 新增 `requirements.txt` 记录当前源码直接依赖的 Python 包。
- 将真实 API Key 保存到本地 `.env`，该文件不进入 Git 跟踪。
- 优化 `train.py` 中的 R-Drop KL 损失归一化、随机种子固定和 AdamW 参数分组，使训练更稳定。
- 新增 `slurm_train.sh`，按默认 GPU 分区、账号和 QOS 提供集群训练入口。
- 将 `requirements.txt` 中 CRF 依赖校准为提供 `torchcrf` 模块的 `pytorch-crf` 包。
- 根据首轮 GPU 结果（验证 F1 约 0.751、测试 F1 约 0.752），新增 BIO 合法转移约束，并将训练参数改为命令行可调。

## Next TODO

- 建议创建 `.venv` 后安装依赖，并验证 `train.py`、`preprocess.py` 和 `main.py` 的基本运行路径。
- 检查 `text.py` 中 `processed/dev.txt` 与当前 `processed/val.txt` 命名不一致的问题。
- 在 GPU 环境中运行训练，记录最佳验证集 F1 和测试集 F1，确认是否达到 80+。
- 拉取最新 `main` 后先运行默认配置（BIO 约束 + `rdrop_alpha=0` + `drop_prob=0.2` + `lr=3e-5`）。
- 如 F1 仍不足 80，优先尝试 `--rdrop-alpha 0.1`、`--drop-prob 0.3`、`--lr 2e-5` 三组对照。

## Open Issues

- 当前仓库初版提交曾包含 IDE 配置、缓存文件和本地 BERT 权重，需要通过新的提交从 Git 跟踪中移除这些本地运行态文件。
- 本地 `.env` 中保存真实 API Key，虽然已被忽略，但历史提交中曾出现明文密钥，建议立即轮换该密钥。
- `bert-base-chinese/pytorch_model.bin` 约 392MB，不适合继续作为普通 Git 文件长期维护。
- 当前尚未在 GPU 上完成新训练配置的完整实验，F1 提升幅度需要实际训练验证。
- 首轮 R-Drop 归一化配置未提升 F1，precision 偏低、recall 偏高，说明模型存在抽取偏多或边界偏宽问题。

## Architecture Decisions

- 保留 `bert-base-chinese` 中的 tokenizer/config 等轻量配置文件；大型模型权重和训练检查点通过 `.gitignore` 排除。
- 不额外引入 `python-dotenv`，由 `main.py` 内置简单 `.env` 读取逻辑，减少依赖面。
- 真实密钥保存在本地 `.env`，仓库只提交 `.env.example` 模板。
- 先保留 BERT + CRF 主架构，通过训练稳定性、损失尺度和优化器配置提升 F1，避免在未验证前引入 BiLSTM 或其它较大结构变更。
- 针对 precision 偏低的问题，优先在 CRF 层加入 BIO 合法转移约束，而不是立即扩大模型结构。
