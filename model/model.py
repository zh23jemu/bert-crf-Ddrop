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
                 labels=None):
        super(BertCRF, self).__init__()

        self.output_size = output_size
        self.labels = labels

        # 加载预训练BERT
        self.bert = AutoModel.from_pretrained(pretrained_path)

        # 获取BERT隐藏层维度
        hidden_size = self.bert.config.hidden_size

        # Dropout
        self.dropout = nn.Dropout(drop_prob)

        # 分类层（直接接BERT输出）
        self.fc = nn.Linear(hidden_size, output_size)

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


    def get_emissions(self, x, attention_mask):
        outputs = self.bert(
            input_ids=x,
            attention_mask=attention_mask
        )
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        emissions = self.fc(sequence_output)
        return emissions

    def forward(self, x, attention_mask=None, labels=None):

        # attention_mask为空时自动生成
        if attention_mask is None:
            attention_mask = (x != 0).long()

        
        emissions = self.get_emissions(x,attention_mask)

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
