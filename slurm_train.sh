#!/bin/bash
#SBATCH --job-name=bert-crf-ddrop
#SBATCH --partition=gpu
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.err

# 用法说明：
# 1. 默认提交到 gpu 分区，避免使用可能额外计费的 aws 分区。
# 2. 如需临时切换分区，可执行：sbatch --partition=gpuHz slurm_train.sh
# 3. 脚本默认使用项目本地 .venv，提交前请先在登录节点或交互节点完成依赖安装。
# 4. 可通过 TRAIN_ARGS 透传训练参数，例如：
#    sbatch --export=ALL,TRAIN_ARGS="--rdrop-alpha 0 --drop-prob 0.2 --lr 3e-5" slurm_train.sh
# 5. 运行CDFA-NER创新模型时，可执行：
#    sbatch --qos=shortjobs --time=01:00:00 --export=ALL,TRAIN_ARGS="--use-discourse-feature" slurm_train.sh

set -euo pipefail

# 切换到提交命令所在目录，确保相对路径如 processed/、model/、image/ 都指向项目内文件。
cd "${SLURM_SUBMIT_DIR}"

# 日志目录属于训练运行态输出，保留在 outputs/logs 方便后续同步分析。
mkdir -p outputs/logs

# 该集群驱动约为 CUDA 12.8 兼容级别；若需要安装 PyTorch GPU 版，推荐使用 cu126 wheel：
# .venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# 使用项目本地虚拟环境运行训练，避免误用系统 Python。
# TRAIN_ARGS 默认空字符串，便于直接 sbatch，也便于提交参数搜索任务。
TRAIN_ARGS=${TRAIN_ARGS:-}
echo "训练附加参数：${TRAIN_ARGS}"
.venv/bin/python train.py ${TRAIN_ARGS}
