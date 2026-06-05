import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import AutoModel


def _is_allowed_transition(from_tag, to_tag):
    """
    判断 BIO 标签之间是否允许转移。

    约束目的：
    1. 禁止从 O 或某类实体直接进入其它类别的 I-XXX，减少解码阶段的非法实体片段。
    2. 禁止 B/I-某类实体后面直接接另一类 I-XXX，避免实体边界被错误延长。
    3. 该约束只作用于 CRF 转移矩阵，不改变标签体系和数据格式。
    """
    if to_tag == "O" or to_tag.startswith("B-"):
        return True
    if not to_tag.startswith("I-"):
        return False

    to_type = to_tag[2:]
    if from_tag.startswith("B-") or from_tag.startswith("I-"):
        return from_tag[2:] == to_type
    return False


class BertCRF(nn.Module):
    def __init__(self,
                 output_size,
                 drop_prob=0.3,
                 pretrained_path="albert-tiny-chinese",
                 labels=None,
                 use_discourse_feature=False,
                 discourse_vocab_size=5,
                 discourse_emb_size=16):
        super(BertCRF, self).__init__()

        self.output_size = output_size
        self.labels = labels
        self.use_discourse_feature = use_discourse_feature

        # 加载预训练BERT
        self.bert = AutoModel.from_pretrained(pretrained_path)

        # 获取BERT隐藏层维度
        hidden_size = self.bert.config.hidden_size

        if self.use_discourse_feature:
            # 教师评语语篇功能嵌入：
            # 将“正向评价/不足语境/建议语境/转折词”等结构化提示映射为低维向量，
            # 再与BERT的上下文表示拼接，突出教师评语“先扬后抑再建议”的领域结构。
            self.discourse_embedding = nn.Embedding(discourse_vocab_size, discourse_emb_size)
            classifier_input_size = hidden_size + discourse_emb_size
        else:
            # 关闭创新特征时保持原始BERT-CRF结构，便于做严格消融对比。
            self.discourse_embedding = None
            classifier_input_size = hidden_size

        # Dropout
        self.dropout = nn.Dropout(drop_prob)

        # 分类层（直接接BERT输出）
        self.fc = nn.Linear(classifier_input_size, output_size)

        # CRF层
        self.crf = CRF(output_size, batch_first=True)

        if labels is not None:
            self.apply_bio_constraints(labels)

    def apply_bio_constraints(self, labels):
        """
        为 CRF 初始化 BIO 合法转移约束。

        torchcrf 中 transitions[i, j] 表示从标签 i 转移到标签 j 的分数。
        对非法 BIO 转移设置较大的负分，训练和解码时都会尽量避开这些路径。
        这类硬约束通常能提升 NER 边界精度，特别适合当前 precision 偏低的结果。
        """
        with torch.no_grad():
            for from_idx, from_tag in enumerate(labels):
                for to_idx, to_tag in enumerate(labels):
                    if not _is_allowed_transition(from_tag, to_tag):
                        self.crf.transitions[from_idx, to_idx] = -10000.0


    def get_emissions(self, x, attention_mask, discourse_ids=None):
        outputs = self.bert(
            input_ids=x,
            attention_mask=attention_mask
        )
        sequence_output = outputs.last_hidden_state

        if self.use_discourse_feature:
            if discourse_ids is None:
                raise ValueError("启用语篇功能特征时必须传入 discourse_ids")
            discourse_output = self.discourse_embedding(discourse_ids)
            sequence_output = torch.cat([sequence_output, discourse_output], dim=-1)

        sequence_output = self.dropout(sequence_output)
        emissions = self.fc(sequence_output)
        return emissions

    def forward(self, x, attention_mask=None, labels=None, discourse_ids=None):

        # attention_mask为空时自动生成
        if attention_mask is None:
            attention_mask = (x != 0).long()

        
        emissions = self.get_emissions(x, attention_mask, discourse_ids=discourse_ids)

        # ======================
        # 训练
        # ======================
        if labels is not None:

            loss = -self.crf(
                emissions,
                labels,
                mask=attention_mask.bool(),
                reduction='mean'
            )

            return loss

        # ======================
        # 推理
        # ======================
        else:

            predictions = self.crf.decode(
                emissions,
                mask=attention_mask.bool()
            )

            return predictions


if __name__ == "__main__":
    pass
