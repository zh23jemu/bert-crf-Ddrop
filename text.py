def count_samples_in_bio_file(file_path):
    """
    统计BIO格式文件中的样本数（每个样本以空行分隔）
    
    参数:
        file_path: BIO格式文件路径
        
    返回:
        样本数量
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            
            # BIO格式中，每个样本以空行分隔（两个换行符）
            if content == "":  # 空文件
                return 0
                
            # 按空行分割样本
            samples = [s.strip() for s in content.split('\n\n') if s.strip()]
            return len(samples)
            
    except FileNotFoundError:
        print(f"错误：文件 {file_path} 未找到")
        return 0
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")
        return 0

# 统计三个数据集的样本数
train_path = "processed/train.txt"
dev_path = "processed/dev.txt"
test_path = "processed/test.txt"

train_samples = count_samples_in_bio_file(train_path)
dev_samples = count_samples_in_bio_file(dev_path)
test_samples = count_samples_in_bio_file(test_path)

# 输出结果
print("=" * 50)
print("数据集样本统计结果")
print("=" * 50)
print(f"训练集 ({train_path}): {train_samples} 条评语")
print(f"验证集 ({dev_path}): {dev_samples} 条评语")
print(f"测试集 ({test_path}): {test_samples} 条评语")
print("-" * 50)
total_samples = train_samples + dev_samples + test_samples
print(f"总计: {total_samples} 条评语")
print()

# 计算比例
print("数据集划分比例:")
print(f"训练集: {train_samples}/{total_samples} ({train_samples/total_samples:.1%})")
print(f"验证集: {dev_samples}/{total_samples} ({dev_samples/total_samples:.1%})")
print(f"测试集: {test_samples}/{total_samples} ({test_samples/total_samples:.1%})")