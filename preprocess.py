import json

entity_types = {
    '性格特质': 'personality',
    '品德行为': 'moral',
    '学习表现': 'study',
    '不足之处': 'weakness',
    '期望建议': 'suggestion'
}


def load_org_dataset(data_path):
    """
    加载原始数据集
    """
    datasets = []
    with open(data_path, "r", encoding="utf-8-sig") as fp:
        for sample in fp.readlines():
            js_data = json.loads(sample)
            datasets.append(js_data)
    return datasets


def to_bio_sequence(datasets, save_path):
    """
    处理为BIO序列
    """
    entity_nums = {k: 0 for k in entity_types.keys()}
    bios = []
    for data in datasets:
        cur_text = data.get('originalText')
        entities = data.get('entities')
        cur_label = ['O'] * len(cur_text)
        for entity in entities:
            # 读取中文标签，并转化为英文标签
            label_type = entity.get('label_type')
            label_type = label_type.strip()
            entity_nums[label_type] += 1
            entity_type = entity_types.get(label_type)
            # 读取实体在文本中的位置
            start_pos = entity.get('start_pos')
            end_pos = entity.get('end_pos')

            # 关键修改：增加索引范围检查
            text_len = len(cur_text)
            if start_pos >= text_len:
                print(f"警告: 实体起始位置超出文本长度 - 文本: '{cur_text}', 起始位置: {start_pos}")
                continue

            # 确保end_pos不超过文本长度
            end_pos = min(end_pos, text_len)

            # 根据实体的位置和类型将标签序列中相应位置该为对应的实体
            cur_label[start_pos] = 'B-' + entity_type
            for idx in range(start_pos + 1, end_pos):
                cur_label[idx] = 'I-' + entity_type

        # 将字符与标签添加到bio序列中，并去掉多余的空格
        for c, l in zip(cur_text, cur_label):
            # 过滤空格
            if c == " ":
                continue
            bios.append(f"{c}\t{l}\n")  # 修改: 使用append而非extend
            # 从句号处分开
            if c in ["。", "；"]:
                bios.append('\n')

    print(entity_nums)
    with open(save_path, "w", encoding="utf-8") as fp:
        fp.writelines(bios)


if __name__ == "__main__":
    # 加载原始数据集
    train_org_path = "comments/comments_train_1.txt"
    train_org_path1 = "comments/comments_train_2.txt"
    test_org_path = "comments/comments_test.jsonl"
    train_dataset = load_org_dataset(train_org_path)
    train_dataset.extend(load_org_dataset(train_org_path1))
    test_dataset = load_org_dataset(test_org_path)
    # 截取训练集的30%作为验证集
    train_size = int(len(train_dataset) * 0.7)
    # 处理为BIO序列
    train_path = "processed/train.txt"
    dev_path = "processed/dev.txt"
    test_path = "processed/test.txt"
    to_bio_sequence(train_dataset[:train_size], train_path)
    to_bio_sequence(train_dataset[train_size:], dev_path)
    to_bio_sequence(test_dataset, test_path)