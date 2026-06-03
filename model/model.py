import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import AutoModel


class BertCRF(nn.Module):
    def __init__(self,
                 output_size,
                 drop_prob=0.3,
                 pretrained_path="albert-tiny-chinese"):
        super(BertCRF, self).__init__()

        self.output_size = output_size

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