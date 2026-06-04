
import os
import random
import argparse
import torch
import warnings
import matplotlib.pyplot as plt
import torch.optim as optim
import numpy as np
from model.model import BertCRF
from seqeval.metrics import classification_report, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from dataset import LABEL, idx2tag, padding, Comments
import torch.nn.functional as F

warnings.filterwarnings('ignore')
# 优先使用GPU，无GPU则用CPU，添加设备日志
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"当前训练设备: {device}")


def set_seed(seed=42):
    """
    固定随机种子，降低同一套参数重复训练时的波动。

    NER 小数据集上的 F1 往往会受到数据打乱、Dropout、CUDA 算子等随机因素影响。
    这里统一固定 Python、NumPy、PyTorch 和 CUDA 的随机源，方便判断一次修改是否真的有效。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(model, train_iter, optimizer, scheduler=None, alpha=0.0):
    """
    训练函数：使用 CRF 损失 + R-Drop KL 正则提升泛化能力。

    关键点：
    1. 两次 forward 保留 Dropout 差异，CRF loss 取平均。
    2. KL loss 只统计真实 token，排除 padding，并按有效 token 数归一化。
    3. alpha 由外部参数控制，便于后续调参，不在函数内部写死。
    """
    model.train()
    train_l_sum, c = 0., 0
    # 用tqdm包装训练迭代器，显示训练进度
    train_bar = tqdm(train_iter, desc="Training")
    for i, (x, attention_mask, y) in enumerate(train_bar):
        x, attention_mask, y = x.to(device), attention_mask.to(device), y.to(device)

        optimizer.zero_grad()
        # loss=model(x, attention_mask=attention_mask, labels=y)  # 确保loss是标量
        if alpha > 0:
            # R-Drop 双forward：仅在 alpha>0 时启用，便于和普通 CRF 训练做对照。
            emissions1 = model.get_emissions(x,attention_mask)
            emissions2 = model.get_emissions(x,attention_mask)
            crf_loss1 = -model.crf(emissions1,y,mask=attention_mask.bool(),reduction='mean')
            crf_loss2 = -model.crf(emissions2,y,mask=attention_mask.bool(),reduction='mean')
            crf_loss = (crf_loss1 + crf_loss2) / 2
            # KL Loss：逐 token 计算两次 forward 的双向 KL，使两个预测分布保持一致。
            p_loss = F.kl_div(
                F.log_softmax(emissions1, dim=-1),
                F.softmax(emissions2, dim=-1),
                reduction='none'
            )
            q_loss = F.kl_div(
                F.log_softmax(emissions2, dim=-1),
                F.softmax(emissions1, dim=-1),
                reduction='none'
            )
            pad_mask = attention_mask.unsqueeze(-1).bool()
            p_loss = p_loss.masked_fill(~pad_mask, 0.)
            q_loss = q_loss.masked_fill(~pad_mask, 0.)
            # 原实现直接 sum 会让 KL 项随 batch 长度膨胀，容易压过 CRF 主损失。
            # 这里按有效 token 数归一化，让 alpha 的量级稳定，通常能改善验证集 F1。
            valid_token_count = pad_mask.sum().clamp_min(1)
            kl_loss = (p_loss.sum() + q_loss.sum()) / (2 * valid_token_count)
            loss = crf_loss + alpha * kl_loss
        else:
            # 强基线：关闭 R-Drop 时只使用标准 CRF 负对数似然。
            # 当前实验显示 R-Drop 配置没有提升 F1，因此默认先回到更稳的普通训练。
            loss = model(x, attention_mask=attention_mask, labels=y)
        loss.backward()
        # 梯度裁剪：防止梯度爆炸（CRF层易出现）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        train_l_sum += loss.item()
        c += 1
        # 进度条实时显示当前loss，替代原打印逻辑，减少日志刷屏
        train_bar.set_postfix({"cur_loss": f"{loss.item():.6f}"})

    avg_loss = train_l_sum / c
    print(f"本epoch平均训练loss: {avg_loss:.6f}")
    return avg_loss


def evaluate(model, eval_iter, report=False, epoch=0):
    """评估函数：核心修复CRF解码逻辑，添加调试日志"""
    model.eval()
    y_true, y_pred = [], []
    total_valid = 0  # 统计有效token数量
    with torch.no_grad():
        eval_bar = tqdm(eval_iter, desc=f"Evaluating Epoch {epoch}")
        for x, attention_mask, y in eval_bar:
            x, attention_mask, y = x.to(device), attention_mask.to(device), y.to(device)

            # 【核心修改1】适配BertBilstmCRF的CRF解码输出
            # 确保模型返回：(loss, 预测标签id序列) 或 仅预测标签id序列（无labels时）
            # 兼容多数CRF实现的输出逻辑：无labels时返回viterbi解码后的标签序列
            predictions = model(x, attention_mask=attention_mask)
            # 额外处理：若模型返回tuple，取第二个元素为预测序列（适配部分CRF实现）
            if isinstance(predictions, tuple):
                predictions = predictions[1]

            # 转换为numpy，过滤padding（保持原有逻辑，优化变量名）
            y_np = y.cpu().numpy()
            mask_np = attention_mask.bool().cpu().numpy()  # mask：1为有效token，0为padding

            for true_seq, pred_seq, mask_seq in zip(y_np, predictions, mask_np):
                               # 有效长度
                valid_len = int(mask_seq.sum())
                # 去掉 CLS 和 SEP
                true_labels = true_seq[1:valid_len - 1]
                pred_labels = pred_seq[1:valid_len - 1]
                # 转标签名
                true_tags = [idx2tag[idx] for idx in true_labels]
                pred_tags = [idx2tag[idx] for idx in pred_labels]
                y_true.append(true_tags)
                y_pred.append(pred_tags)

    # F1
    eval_f1 = f1_score(y_true, y_pred)
    print(f"\nEval F1: {eval_f1:.6f}")
    if report:

        print("\n===== Classification Report =====")

        print(classification_report(
            y_true,
            y_pred,
            zero_division=0
        ))
    return eval_f1



def main(args):
    set_seed(args.get('seed', 42))

    # 1. 加载数据集（保持原有路径，添加加载日志）
    print("===== 开始加载数据集 =====")
    train_set = Comments("processed/train.txt", args['pretrained_path'])
    dev_set = Comments("processed/val.txt", args['pretrained_path'])
    test_set = Comments("processed/test.txt", args['pretrained_path'])
    print(f"训练集样本数: {len(train_set)}, 验证集: {len(dev_set)}, 测试集: {len(test_set)}")

    # 2. 构建DataLoader【修改】统一配置，关闭persistent_workers避免内存泄漏
    # collate_fn=padding：确保你的padding函数返回(x, attention_mask, y)，顺序不可变
    def build_dataloader(dataset, batch_size, is_train=False):
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            collate_fn=padding,
            shuffle=is_train,  # 训练集打乱，验证/测试集不打乱
            num_workers=0,  # 新手友好：设为0避免多进程数据加载错误（Windows/Linux通用）
            pin_memory=True if torch.cuda.is_available() else False
        )

    train_iter = build_dataloader(train_set, args['batch_size'], is_train=True)
    dev_iter = build_dataloader(dev_set, args['batch_size'])
    test_iter = build_dataloader(test_set, args['batch_size'])

    # 3. 初始化模型（保持原有参数，添加模型日志）
    print("\n===== 初始化模型 =====")
    model = BertCRF(
        output_size=args['output_size'],
        drop_prob=args['drop_prob'],
        pretrained_path=args['pretrained_path'],
        labels=LABEL
    )
    model.to(device)
    # 打印模型参数量（粗略统计）
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数量: {total_params/1e6:.2f}M, 可训练参数量: {trainable_params/1e6:.2f}M")

    # 4. 优化器（BERT-CRF推荐配置）
    print("\n===== 配置优化器和学习率调度器 =====")
    # BERT 参数分组：bias 和 LayerNorm 不做权重衰减，避免破坏预训练层的归一化参数。
    # 这是 BERT 微调的常见配置，对小样本 NER 通常比整层统一 weight_decay 更稳。
    no_decay = ["bias", "LayerNorm.weight"]
    bert_decay_params = []
    bert_no_decay_params = []
    for name, param in model.bert.named_parameters():
        if any(nd in name for nd in no_decay):
            bert_no_decay_params.append(param)
        else:
            bert_decay_params.append(param)

    optimizer = optim.AdamW([
        # BERT主体：小学习率，带权重衰减
        {
            'params': bert_decay_params,
            'lr': args['lr'],
            'weight_decay': args['weight_decay']
        },

        # BERT中的bias/LayerNorm：小学习率，不做权重衰减
        {
            'params': bert_no_decay_params,
            'lr': args['lr'],
            'weight_decay': 0.0
        },

        # 全连接层：大学习率，帮助新初始化分类头更快适配标签空间
        {
            'params': model.fc.parameters(),
            'lr': args['head_lr'],
            'weight_decay': 0.0
        },

        # CRF层：大学习率，帮助转移矩阵更快学习BIO约束
        {
            'params': model.crf.parameters(),
            'lr': args['head_lr'],
            'weight_decay': 0.0
        }

    ], betas=(0.9, 0.999))

    # 5. 学习率调度器
    total_steps = len(train_iter) * args['epochs']

    # warmup步数（10%）
    warmup_steps = int(total_steps * 0.1)
    # warmup scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # 6. 训练初始化【核心修改4】处理best_model为None的情况，添加早停机制（可选）
    best_f1 = 0.0
    best_model_path="model/best_model.pth"  # 最佳模型保存路径
    train_losses, eval_f1s = [], []
    early_stop_count = 0  # 早停计数器：连续5轮F1不提升则停止训练
    early_stop_patience = args['early_stop_patience']

    # 7. 开始训练
    print(f"\n===== 开始训练（共{args['epochs']}个epoch） =====")
    for e in range(args['epochs']):
        current_epoch = e + 1
        print(f"\n========== Epoch {current_epoch}/{args['epochs']} ==========")
        # 训练
        train_loss = train(model, train_iter, optimizer, scheduler, alpha=args['rdrop_alpha'])
        # 评估
        eval_f1 = evaluate(model, dev_iter, epoch=current_epoch)
        # 记录loss和F1
        train_losses.append(train_loss)
        eval_f1s.append(eval_f1)
        # 获取当前学习率
        current_lr = scheduler.get_last_lr()[0]

        # 保存最佳模型（F1提升时）
        if eval_f1 > best_f1:
            best_f1 = eval_f1
            torch.save(model.state_dict(), best_model_path)
            print(f"保存最佳模型！当前最佳F1: {best_f1:.6f}")
            early_stop_count = 0  # 重置早停计数器
        else:
            early_stop_count += 1
            print(f"F1未提升，早停计数器: {early_stop_count}/{early_stop_patience}")

        # 打印本轮训练信息
        print(f"Epoch {current_epoch} | Train Loss: {train_loss:.6f} | Val F1: {eval_f1:.6f} | LR: {current_lr:.2e} | Best F1: {best_f1:.6f}")

        # 早停：连续多轮F1不提升，提前终止训练（避免过拟合）
        if early_stop_count >= early_stop_patience:
            print(f"连续{early_stop_patience}轮F1未提升，触发早停！")
            break

    # 8. 最终测试（使用最佳模型）
    print("\n===== 使用最佳模型测试 =====")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_f1 = evaluate(model, test_iter, report=True)
    print(f"测试集最终F1值: {test_f1:.6f}")
    print(f"训练过程中最佳验证F1值: {best_f1:.6f}")

    # 9. 绘制训练曲线
    print("\n===== Plot Training Curves =====")

    import matplotlib
    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    epochs_range = range(1, len(train_losses) + 1)

    # 创建画布
    _, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    # ================= Loss Curve =================
    axes[0].plot(
        epochs_range,
        train_losses,
        linewidth=2
    )

    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True)

    # ================= F1 Curve =================
    axes[1].plot(
        epochs_range,
        eval_f1s,
        linewidth=2,
        label="Validation F1"
    )

    axes[1].axhline(
        y=best_f1,
        linestyle="--",
        linewidth=1.5,
        label=f"Best F1 = {best_f1:.4f}"
    )

    axes[1].set_title("Validation F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("F1 Score")
    axes[1].legend()
    axes[1].grid(True)

    # 自动布局
    plt.tight_layout()

    # 保存图片
    plt.savefig(
        "image/train_curve.png",
        bbox_inches="tight"
    )

    # 关闭画布
    plt.close()

    print("训练曲线已保存：image/train_curve.png")


def parse_args():
    """
    解析训练参数，便于在 Slurm 上直接做小规模参数搜索。

    示例：
    .venv/bin/python train.py --seed 21 --rdrop-alpha 0 --drop-prob 0.25 --lr 2.8e-5
    """
    parser = argparse.ArgumentParser(description="训练 BERT-CRF 教师评语实体抽取模型")
    parser.add_argument("--pretrained-path", default="bert-base-chinese", help="本地预训练BERT目录")
    parser.add_argument("--lr", type=float, default=2.8e-5, help="BERT主体学习率")
    parser.add_argument("--head-lr", type=float, default=2e-4, help="分类层和CRF层学习率")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="BERT权重衰减")
    parser.add_argument("--rdrop-alpha", type=float, default=0.0, help="R-Drop KL损失权重，0表示关闭")
    parser.add_argument("--seed", type=int, default=21, help="随机种子")
    parser.add_argument("--batch-size", type=int, default=16, help="训练批大小")
    parser.add_argument("--epochs", type=int, default=20, help="最大训练轮数")
    parser.add_argument("--drop-prob", type=float, default=0.25, help="Dropout比例")
    parser.add_argument("--early-stop-patience", type=int, default=10, help="验证F1连续不提升的早停轮数")
    cli_args = parser.parse_args()

    # 转成原 main 函数使用的字典结构，保持主体代码改动最小。
    return {
        'pretrained_path': cli_args.pretrained_path,
        "lr": cli_args.lr,
        "head_lr": cli_args.head_lr,
        "weight_decay": cli_args.weight_decay,
        "rdrop_alpha": cli_args.rdrop_alpha,
        "seed": cli_args.seed,
        "batch_size": cli_args.batch_size,
        "epochs": cli_args.epochs,
        "output_size": len(LABEL),
        "drop_prob": cli_args.drop_prob,
        "early_stop_patience": cli_args.early_stop_patience
    }


if __name__ == "__main__":
    # 优化训练参数，适配NER任务。
    # 默认配置先使用 BIO 约束 + 普通 CRF 训练，目标是提高 precision 并恢复到更强基线。
    params = parse_args()
    print("===== 训练参数配置 =====")
    for k, v in params.items():
        print(f"{k}: {v}")
    # 启动训练
    main(params)
