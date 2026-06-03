from collections import defaultdict

# 实体类别映射（中文→英文，与原始定义一致）
entity_types = {
    '品德行为': 'moral',
    '性格特质': 'personality',
    '学习表现': 'study',
    '不足之处': 'weakness',
    '期望建议': 'suggestion'
}
# 英文→中文反向映射（用于从B标签提取中文类别）
reverse_map = {v: k for k, v in entity_types.items()}
# 输出顺序（固定为指定顺序）
output_order = ['品德行为', '性格特质', '学习表现', '不足之处', '期望建议']


def count_bio_entities(bio_file_path):
    """
    从BIO文件中统计实体数量（每个实体按B标签计数1次）
    :param bio_file_path: BIO格式文件路径（如train.txt）
    :return: 按中文类别统计的实体数量字典
    """
    # 初始化计数字典（确保所有类别都有记录，默认0）
    entity_counts = {category: 0 for category in entity_types.keys()}
    # 记录当前句子中已统计的实体（避免同一实体的B标签被重复计数）
    current_sentence_entities = set()
    
    with open(bio_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # 空行表示句子结束，重置当前句子的实体记录
            if not line:
                current_sentence_entities.clear()
                continue
            
            # 按\t分割字符和标签（BIO文件格式：字符\t标签）
            parts = line.split('\t')
            if len(parts) < 2:
                continue  # 跳过格式异常的行
            tag = parts[1]
            
            # 只处理B标签（B标签代表实体的开始，每个B对应一个实体）
            if tag.startswith('B-'):
                # 提取英文实体类型（如从B-moral中提取moral）
                eng_type = tag.split('-')[1]
                # 映射到中文类别（如moral→品德行为）
                ch_type = reverse_map.get(eng_type)
                
                # 确保中文类别有效，且当前句子中未统计过该实体（避免重复）
                if ch_type in entity_counts and ch_type not in current_sentence_entities:
                    entity_counts[ch_type] += 1
                    current_sentence_entities.add(ch_type)
    
    return entity_counts


if __name__ == "__main__":
    # 人工检查后的BIO文件路径（根据实际路径修改）
    bio_files = {
        "训练集": "processed/train.txt",
        "验证集": "processed/val.txt",
        "测试集": "processed/test.txt"
    }
    
    # 遍历文件并统计
    for dataset_name, file_path in bio_files.items():
        print(f"===== {dataset_name} 实体数量统计 =====")
        counts = count_bio_entities(file_path)
        # 按指定顺序输出
        for category in output_order:
            print(f"{category}: {counts[category]}")
        print()  # 空行分隔不同数据集