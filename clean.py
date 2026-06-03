import re
import pandas as pd
import jieba
from typing import List, Optional, Tuple


class CommentCleaner:
    def __init__(self, stopwords_path: Optional[str] = None):
        """
        初始化清洗器：支持停用词、重复行检测、全流程清洗（保留原始格式）
        :param stopwords_path: 自定义停用词路径（可选）
        """
        # 停用词加载（内置默认 + 自定义，本版本默认不过滤停用词）
        self.stopwords = self._load_default_stopwords()
        if stopwords_path:
            self.stopwords.update(self._load_custom_stopwords(stopwords_path))

    def _load_default_stopwords(self) -> set:
        """加载针对教师评语优化的默认停用词（暂不用于过滤）"""
        return {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个',
            '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看',
            '好', '自己', '这', '那', '啊', '吧', '把', '被', '比', '别', '才', '朝',
            # 可根据需求补充/精简停用词，实际暂不用
        }

    def _load_custom_stopwords(self, file_path: str) -> set:
        """从文件加载自定义停用词（暂不用于过滤）"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return set([line.strip() for line in f if line.strip()])
        except FileNotFoundError:
            print(f"警告：自定义停用词文件 {file_path} 未找到，跳过加载")
            return set()

    def clean_single_comment(self, text: str, min_length: int = 1) -> str:
        """
        单条评语清洗核心逻辑：去噪声→（不删除停用词、不分词后空格拼接）→语句规整
        :param text: 原始评语文本（保留原始格式）
        :param min_length: 保留词语的最小长度（这里实际暂未用到分词过滤逻辑）
        :return: 清洗后文本（保留原始格式、停用词，语句不隔开）
        """
        if not isinstance(text, str) or not text.strip():
            return ""

        # 1. 移除噪声（HTML/URL/特殊符号等）
        text = self._remove_noise(text)

        # 2. 文本归一化（修复标点、简化表达，让语句更通顺）
        text = self._normalize_text(text)
        return text

    def _remove_noise(self, text: str) -> str:
        """移除文本中的噪声内容（保留原始格式符号）"""
        # 移除HTML标签
        text = re.sub(r'<[^>]+>', '', text)
        # 移除URL
        text = re.sub(r'http\S+|www\S+', '', text)
        # 移除邮箱
        text = re.sub(r'\S+@\S+', '', text)
        # 移除特殊符号（保留中文、基本标点，可按需调整）
        text = re.sub(r'[^\w\s\u4e00-\u9fff，。！？；：“”‘’《》〔〕【】、,.!?;:"]', '', text)
        # 移除数字（如需保留可删除）
        text = re.sub(r'\d+', '', text)
        # 移除多余空格、换行（保留原始换行符）
        text = re.sub(r' +', ' ', text).strip()  # 仅处理空格，保留换行
        return text

    def _normalize_text(self, text: str) -> str:
        """文本归一化：修复标点、简化表达，让语句更通顺"""
        # 修复标点（中文标点统一、删除多余标点）
        text = re.sub(r'，+', '，', text)  # 合并连续逗号
        text = re.sub(r'。+', '。', text)  # 合并连续句号
        # 简化表达（示例：“希”→“希望”，根据场景扩展）
        replace_map = {"希": "希望", "需": "需要", "应": "应该"}
        for old, new in replace_map.items():
            text = text.replace(old, new)
        # 移除多余空格（保留原始换行符）
        text = re.sub(r' +', ' ', text).strip()
        return text

    def process_missing_values(self, comments: List[str]) -> List[str]:
        """
        缺失值处理：过滤空行（保留原始非空行）
        :param comments: 原始评语列表（每行一条）
        :return: 过滤空行后的列表
        """
        return [c for c in comments if c.strip()]

    def detect_duplicates(self, comments: List[str]) -> List[bool]:
        """
        重复行检测：标记重复行（基于清洗后内容）
        :param comments: 原始评语列表
        :return: 标记列表（True=重复行，False=唯一行）
        """
        cleaned_comments = [self.clean_single_comment(c) for c in comments]
        seen = set()
        is_duplicate = []
        for c in cleaned_comments:
            if c in seen:
                is_duplicate.append(True)
            else:
                seen.add(c)
                is_duplicate.append(False)
        return is_duplicate

    def process_duplicates(self, comments: List[str],
                           action: str = "mark") -> Tuple[List[str], List[bool]]:
        """
        重复值处理：
        - action="mark"：标记重复行（返回标记列表）
        - action="remove"：去重（保留第一条）
        :param comments: 原始评语列表
        :param action: 处理方式（"mark" 或 "remove"）
        :return: 处理后的评语列表 + 重复标记（或空列表）
        """
        if action == "mark":
            is_duplicate = self.detect_duplicates(comments)
            return comments, is_duplicate
        elif action == "remove":
            cleaned_comments = [self.clean_single_comment(c) for c in comments]
            seen = set()
            result = []
            for c, cleaned in zip(comments, cleaned_comments):
                if cleaned not in seen:
                    seen.add(cleaned)
                    result.append(c)
            return result, []
        else:
            raise ValueError("action 仅支持 'mark' 或 'remove'")

    def process_outliers(self, comments: List[str], len_threshold: int = 5, keep_outliers: bool = True) -> List[str]:
        """
        异常值处理：
        - keep_outliers=True：保留所有行（仅清洗内容）
        - keep_outliers=False：过滤极短文本
        :param comments: 原始评语列表
        :param len_threshold: 文本最小长度（清洗前）
        :param keep_outliers: 是否保留异常行
        :return: 处理后的评语列表
        """
        if keep_outliers:
            return comments  # 仅清洗内容，保留原始异常行
        else:
            return [
                c for c in comments
                if len(c.strip()) >= len_threshold and "乱码" not in c
            ]

    def full_pipeline(self, input_path: str, output_path: str,
                      keep_duplicates: bool = True, keep_outliers: bool = True,
                      duplicate_action: str = "mark") -> Tuple[List[str], List[bool]]:
        """
        全流程清洗管道：
        1. 读取数据 → 2. 缺失值处理 → 3. 重复值处理（标记/去重） →
        4. 异常值处理 → 5. 文本清洗 → 6. 保存结果（保留原始格式）
        :param input_path: 输入文件路径（txt格式，每行一条评语）
        :param output_path: 输出文件路径（保留原始格式）
        :param keep_duplicates: 是否保留重复行（仅当 duplicate_action="remove" 时失效）
        :param keep_outliers: 是否保留异常行（极短文本）
        :param duplicate_action: 重复行处理方式（"mark" 或 "remove"）
        :return: 清洗后的评语列表 + 重复标记列表（或空）
        """
        # 1. 读取数据（按行读取，保留原始格式）
        with open(input_path, 'r', encoding='utf-8') as f:
            comments = [line.rstrip('\n') for line in f]  # 保留原始换行符

        # 2. 缺失值处理（过滤空行）
        comments = self.process_missing_values(comments)

        # 3. 重复值处理（标记或去重）
        comments, is_duplicate = self.process_duplicates(comments, action=duplicate_action)

        # 4. 异常值处理（可选保留异常行）
        comments = self.process_outliers(comments, keep_outliers=keep_outliers)

        # 5. 文本清洗（逐行清洗，保留原始格式、停用词，语句不隔开）
        cleaned_comments = [
            self.clean_single_comment(c) if c.strip() else ""
            for c in comments
        ]

        # 6. 保存结果（保留原始行顺序和格式）
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(cleaned_comments))  # 每行对应原始行的清洗结果

        print(f"清洗完成！结果已保存至 {output_path}")
        print(f"原始行数：{len(comments)} → 清洗后行数：{len(cleaned_comments)}")
        return cleaned_comments, is_duplicate


# ------------------------  使用示例  ------------------------
if __name__ == "__main__":
    # 1. 配置路径（根据实际文件调整）
    input_file = "comments/comments.txt"  # 原始评语文本文件（每行一条）
    output_file = "comments/comments_1.txt"  # 清洗后输出文件，生成一个结果文件

    # 2. 初始化清洗器（可传入自定义停用词路径，实际暂不用于过滤停用词）
    cleaner = CommentCleaner(stopwords_path="custom_stopwords.txt")

    # 3. 执行全流程清洗（保留重复行并标记或直接去重，根据需求选择 duplicate_action）
    # 若要去重后生成一个文件，设置 duplicate_action="remove"
    cleaned_result, is_duplicate = cleaner.full_pipeline(
        input_path=input_file,
        output_path=output_file,
        keep_duplicates=True,  # 保留重复行（仅标记，实际去重由 duplicate_action 控制，若设置为 remove 则此参数失效）
        keep_outliers=True,  # 保留异常行（如“短短”）
        duplicate_action="remove"  # 这里设置为 remove 执行去重，若想标记重复行可设为 mark
    )

    # 4. 打印相关结果（可选）
    print("\n清洗后示例（前几条，保留原始格式、停用词，语句不隔开）：")
    for i, c in enumerate(cleaned_result[:4]):
        print(f"[{i + 1}] {c}")