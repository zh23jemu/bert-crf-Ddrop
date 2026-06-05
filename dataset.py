import torch
import torch.utils.data as data
from transformers import BertTokenizerFast

# 定义NER标签体系（O为非实体，B-XXX为实体开头，I-XXX为实体内部）
LABEL = ("O", "B-personality", "I-personality",
         "B-moral", "I-moral", "B-study", "I-study",
         "B-weakness", "I-weakness",
         "B-suggestion", "I-suggestion")

# 标签<->索引映射（核心：0对应O标签，不可修改）
tag2idx = {tag: idx for idx, tag in enumerate(LABEL)}
idx2tag = {idx: tag for idx, tag in enumerate(LABEL)}
print(f"标签映射关系: {tag2idx}")
print(f"标签总数: {len(LABEL)}")

# 评语语篇功能标签体系：
# neutral：默认背景；positive_eval：前置正向评价；weakness_context：不足/问题语境；
# suggestion_context：期望/建议语境；transition_context：转折触发词本身。
DISCOURSE_LABEL = ("neutral", "positive_eval", "weakness_context",
                   "suggestion_context", "transition_context")
discourse2idx = {tag: idx for idx, tag in enumerate(DISCOURSE_LABEL)}
idx2discourse = {idx: tag for idx, tag in enumerate(DISCOURSE_LABEL)}

# 教师评语常见功能触发词。该特征只依赖原始文本，不使用BIO标签，避免验证/测试阶段标签泄漏。
WEAKNESS_TRIGGERS = ("不足之处", "但是", "不过", "只是", "需要", "有待", "缺乏", "不够", "但", "需")
SUGGESTION_TRIGGERS = ("希望", "建议", "若能", "以后", "今后", "应该", "争取", "望", "应", "要")
TRANSITION_TRIGGERS = ("但是", "不过", "只是", "但")
POSITIVE_CUES = (
    "优秀", "聪明", "活泼", "友善", "尊敬", "关心", "认真", "负责", "端正", "突出",
    "擅长", "热情", "乐于助人", "遵守", "团结", "勤奋", "积极", "进步", "开朗",
    "自信", "踏实", "细致", "诚恳", "有礼", "良好", "按时", "专心", "敏捷",
    "工整", "清秀", "品学兼优", "思维", "努力", "稳重", "善良", "懂事"
)
CLAUSE_DELIMITERS = set("，。！？；；,.!?;：:")


def split_clauses(text):
    """
    按教师评语中常见标点切分分句，返回每个分句在原文本中的[start, end)范围。

    教师评语通常由“优点描述—转折不足—建议期望”多个短分句组成，分句粒度比整句更适合
    表达语篇功能。标点本身包含在前一个分句中，避免位置错位。
    """
    clauses = []
    start = 0
    for idx, char in enumerate(text):
        if char in CLAUSE_DELIMITERS:
            if start < idx + 1:
                clauses.append((start, idx + 1))
            start = idx + 1
    if start < len(text):
        clauses.append((start, len(text)))
    return clauses


def contains_any(text, triggers):
    """判断文本片段中是否出现任一触发词。"""
    return any(trigger in text for trigger in triggers)


def find_non_overlapping_spans(text, triggers):
    """
    查找触发词位置，按长词优先避免“但是”又被重复匹配成“但”。

    返回值为[(start, end), ...]，用于把转折词本身标为 transition_context。
    """
    spans = []
    occupied = [False] * len(text)
    for trigger in sorted(triggers, key=len, reverse=True):
        start = text.find(trigger)
        while start != -1:
            end = start + len(trigger)
            if not any(occupied[start:end]):
                spans.append((start, end))
                for idx in range(start, end):
                    occupied[idx] = True
            start = text.find(trigger, start + 1)
    return spans


def build_discourse_ids(words):
    """
    为每个字符生成教师评语语篇功能ID。

    规则特点：
    1. 只使用文本触发词、分句位置和积极评价词，不读取BIO标签，避免数据泄漏。
    2. 出现不足/建议触发词的当前分句分别标为 weakness_context / suggestion_context。
    3. 转折词本身额外标为 transition_context，帮助模型捕捉“但/不过/只是”后的语义切换。
    4. 首个不足/建议分句之前，含积极评价词的分句标为 positive_eval，贴合教师评语先扬后抑结构。
    """
    text = "".join(words)
    discourse_ids = [discourse2idx["neutral"]] * len(text)
    clauses = split_clauses(text)
    clause_roles = []
    first_shift_start = len(text)

    for start, end in clauses:
        clause_text = text[start:end]
        # 先判断不足语境，再判断建议语境，避免“需要改进”被单字“要”误判为建议。
        if contains_any(clause_text, WEAKNESS_TRIGGERS):
            role = "weakness_context"
            first_shift_start = min(first_shift_start, start)
        elif contains_any(clause_text, SUGGESTION_TRIGGERS):
            role = "suggestion_context"
            first_shift_start = min(first_shift_start, start)
        else:
            role = "neutral"
        clause_roles.append((start, end, role))

    for start, end, role in clause_roles:
        clause_text = text[start:end]
        if role == "neutral" and start < first_shift_start and contains_any(clause_text, POSITIVE_CUES):
            role = "positive_eval"
        role_idx = discourse2idx[role]
        for idx in range(start, end):
            discourse_ids[idx] = role_idx

    for start, end in find_non_overlapping_spans(text, TRANSITION_TRIGGERS):
        for idx in range(start, end):
            discourse_ids[idx] = discourse2idx["transition_context"]

    return discourse_ids


class Comments(data.Dataset):
    def __init__(self, data_path, pretrained_path, maxlen=256):
        super().__init__()
        self.maxlen = maxlen  # 句子最大长度（包含[CLS]和[SEP]）
        self.pretrained_path = pretrained_path
        # 【优化1】先加载tokenizer，再构建数据集
        self.tokenizer = BertTokenizerFast.from_pretrained(pretrained_path)
        # 特殊token的ID（从tokenizer中直接获取，避免硬编码）
        self.cls_id = self.tokenizer.cls_token_id
        self.sep_id = self.tokenizer.sep_token_id
        self.pad_id = self.tokenizer.pad_token_id
        # 构建数据集
        self.dataset = self.buildingDataset(data_path)
        # 打印数据集信息，验证加载是否正确
        print(f"从{data_path}加载到{len(self.dataset)}条有效样本")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        words, tags = self.dataset[index]
        input_ids, discourse_ids, label_ids = [], [], []
        char_discourse_ids = build_discourse_ids(words)

        # 处理每个词和对应的标签，适配subword切分
        for word, tag, discourse_id in zip(words, tags, char_discourse_ids):
            # 将词切分为subword并转换为id（不添加特殊token）
            sub_tokens = self.tokenizer.tokenize(word)
            if not sub_tokens:  # 过滤空词
                continue
            sub_token_ids = self.tokenizer.convert_tokens_to_ids(sub_tokens)
            input_ids.extend(sub_token_ids)
            discourse_ids.extend([discourse_id] * len(sub_tokens))

            # 【核心修正2】仅对B-开头的标签做subword扩展，O/I-标签直接重复
            tag_idx = tag2idx[tag]
            if tag.startswith("B-"):
                # B-XXX标签：第一个subword保留B-，后续转为I-
                i_tag = tag.replace("B-", "I-")
                i_tag_idx = tag2idx.get(i_tag, tag_idx)  # 防止无对应I-标签
                label_ids.append(tag_idx)
                label_ids.extend([i_tag_idx] * (len(sub_tokens) - 1))
            else:
                # O/I-XXX标签：所有subword使用相同标签
                label_ids.extend([tag_idx] * len(sub_tokens))

            # 提前截断，避免后续处理过长
            if len(input_ids) >= self.maxlen - 2:
                break

        # 【核心修正3】添加[CLS]和[SEP]，并截断到maxlen（严格控制长度）
        # 预留2个位置给[CLS]和[SEP]
        input_ids = input_ids[:self.maxlen - 2]
        discourse_ids = discourse_ids[:self.maxlen - 2]
        label_ids = label_ids[:self.maxlen - 2]
        # 拼接特殊token：[CLS] + 正文 + [SEP]
        input_ids = [self.cls_id] + input_ids + [self.sep_id]
        # 特殊token没有实际语篇功能，统一置为neutral。
        discourse_ids = [discourse2idx["neutral"]] + discourse_ids + [discourse2idx["neutral"]]
        # 特殊token的标签为O（idx=0），符合NER标注规范
        label_ids = [0] + label_ids + [0]

        # 校验长度（防止空序列）
        assert len(input_ids) == len(label_ids), f"输入和标签长度不一致：{len(input_ids)} vs {len(label_ids)}"
        assert len(input_ids) == len(discourse_ids), f"输入和语篇功能长度不一致：{len(input_ids)} vs {len(discourse_ids)}"
        assert len(input_ids) <= self.maxlen, f"输入长度超过maxlen：{len(input_ids)}"

        return input_ids, discourse_ids, label_ids

    def buildingDataset(self, data_path):
        """
        构建数据集：加载txt文件，按行解析词-标签对，处理格式错误
        数据格式：每行 词\t标签，句子之间用空行分隔
        """
        dataset = []
        with open(data_path, "r", encoding="utf-8") as fp:
            lines = fp.readlines()
            cur_words, cur_tags = [], []
            for line_num, line in enumerate(lines, 1):  # 行号从1开始，方便定位错误
                line = line.strip()
                if not line:  # 空行表示句子结束
                    if cur_words and cur_tags:  # 仅添加非空句子
                        dataset.append([cur_words, cur_tags])
                    cur_words, cur_tags = [], []
                    continue
                # 解析词-标签对，处理格式错误
                try:
                    word, tag = line.split("\t")
                    # 校验标签是否在定义的LABEL中，过滤无效标签
                    if tag not in tag2idx:
                        print(f"第{line_num}行：无效标签{tag}，已替换为O")
                        tag = "O"
                    cur_words.append(word)
                    cur_tags.append(tag)
                except ValueError:
                    print(f"第{line_num}行格式错误：{line}，已跳过")
                    continue
            # 处理文件末尾的最后一个句子
            if cur_words and cur_tags:
                dataset.append([cur_words, cur_tags])

        # 【核心修正5】删除无意义的排序，避免批次内长度差异过大
        # self.dataset.sort(key=lambda x: len(x[0]))  # 注释/删除此行

        return dataset


def padding(batch):
    """
    批次填充函数：将batch内的样本填充到相同长度
    返回：input_ids(LongTensor)、discourse_ids(LongTensor)、attention_mask(LongTensor)、labels(LongTensor)
    attention_mask：1.0表示有效token，0.0表示padding
    """
    # 获取batch内最长样本的长度
    max_seq_len = max([len(sample[0]) for sample in batch])
    input_ids, discourse_ids, labels, attention_masks = [], [], [], []

    for sample in batch:
        seq_len = len(sample[0])
        pad_len = max_seq_len - seq_len

        # 填充输入序列（PAD_ID=0）
        cur_input = sample[0] + [0] * pad_len
        # 语篇功能padding部分使用neutral，避免给模型注入虚假上下文。
        cur_discourse = sample[1] + [discourse2idx["neutral"]] * pad_len
        # 填充标签序列（O标签=0，padding部分也标为O）
        cur_label = sample[2] + [0] * pad_len
        # 生成attention_mask：有效token为1.0，padding为0.0（FloatTensor）
        cur_mask = [1.0] * seq_len + [0.0] * pad_len

        input_ids.append(cur_input)
        discourse_ids.append(cur_discourse)
        labels.append(cur_label)
        attention_masks.append(cur_mask)

    # 转换为Tensor，符合PyTorch规范
    input_ids_tensor = torch.LongTensor(input_ids)
    discourse_ids_tensor = torch.LongTensor(discourse_ids)
    labels_tensor = torch.LongTensor(labels)
    attention_mask_tensor = torch.LongTensor(attention_masks) 

    return input_ids_tensor, discourse_ids_tensor, attention_mask_tensor, labels_tensor


if __name__ == "__main__":
    # 测试数据集加载和padding函数
    train_path = "processed/val.txt"  # 替换为你的数据路径
    pretrained_path = "bert-base-chinese"
    maxlen = 256

    try:
        # 加载数据集
        train_set = Comments(train_path, pretrained_path, maxlen)
        if len(train_set) == 0:
            raise ValueError("数据集为空，请检查数据路径和格式")
        # 测试单条样本
        idx = 0
        x, discourse_ids, y = train_set[idx]
        print(f"\n第{idx+1}条样本测试：")
        print(f"输入序列长度: {len(x)}")
        print(f"语篇功能序列长度: {len(discourse_ids)}")
        print(f"标签序列长度: {len(y)}")
        print(f"输入序列前10个token: {x[:10]}")
        print(f"语篇功能前10个idx: {discourse_ids[:10]}")
        print(f"语篇功能前10个名称: {[idx2discourse[i] for i in discourse_ids[:10]]}")
        print(f"标签序列前10个idx: {y[:10]}")
        print(f"标签序列前10个名称: {[idx2tag[i] for i in y[:10]]}")

        # 测试DataLoader和padding
        batch_size = 2
        train_iter = data.DataLoader(
            dataset=train_set,
            batch_size=batch_size,
            collate_fn=padding,
            shuffle=True,
            num_workers=0
        )
        # 测试单个batch
        for batch_idx, (x_batch, discourse_batch, mask_batch, y_batch) in enumerate(train_iter):
            print(f"\n第{batch_idx+1}个Batch测试：")
            print(f"input_ids shape: {x_batch.shape}")  # [batch_size, max_seq_len]
            print(f"discourse_ids shape: {discourse_batch.shape}")  # [batch_size, max_seq_len]
            print(f"attention_mask shape: {mask_batch.shape}")  # [batch_size, max_seq_len]
            print(f"labels shape: {y_batch.shape}")  # [batch_size, max_seq_len]
            print(f"attention_mask示例: {mask_batch[0].numpy()}")
            print(f"有效token数: {mask_batch[0].sum().item()}")
            print(f"填充后长度: {mask_batch[0].shape[0]}")
            break  # 仅测试1个batch

        print("\n数据集和padding函数测试通过！")

    except Exception as e:
        print(f"\n测试出错: {e}")
        import traceback
        traceback.print_exc()
