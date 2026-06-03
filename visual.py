import matplotlib.pyplot as plt
import numpy as np

train_entity_nums = {
    '品德行为': 399,
    '性格特质': 420,
    '学习表现': 432,
    '不足之处': 338,
    '期望建议': 502
}

dev_entity_nums = {
    '品德行为': 178,
    '性格特质': 143,
    '学习表现': 186,
    '不足之处': 154,
    '期望建议': 225
}

test_entity_nums = {
    '品德行为': 180,
    '性格特质':201,
    '学习表现': 202,
    '不足之处': 126,
    '期望建议': 237
}

# 获取实体类别名称
labels = list(train_entity_nums.keys())
x = np.arange(len(labels))
y_train = list(train_entity_nums.values())
y_dev = list(dev_entity_nums.values())
y_test = list(test_entity_nums.values())
bar_width = 0.3
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.bar(x, y_train, width=bar_width)
plt.bar(x + bar_width, y_dev, width=bar_width)
plt.bar(x + 2 * bar_width, y_test, width=bar_width)
plt.xlabel('实体类别')
plt.ylabel('数量')
plt.legend(['训练集', '验证集', '测试集'])
# 设置横轴刻度标签
plt.xticks(x, labels)
plt.savefig("image/label_distribution.png")
plt.show()
