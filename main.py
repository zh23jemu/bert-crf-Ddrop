import json
import re
import os
from typing import List, Dict, Tuple
from rich.status import Status
from rich.progress import track

from openai import OpenAI


def load_env_file(env_path: str = ".env") -> None:
    """
    从本地 .env 文件加载运行配置。

    设计说明：
    1. .env 会被 .gitignore 忽略，用于保存 API Key 这类本地敏感配置。
    2. 如果系统环境变量中已经存在同名配置，则不覆盖，便于服务器或 CI 直接注入变量。
    3. 这里只实现当前项目需要的简单 KEY=VALUE 解析，避免额外引入 python-dotenv 依赖。
    """
    if not os.path.exists(env_path):
        return

    with open(env_path, "r", encoding="utf-8") as fp:
        for raw_line in fp:
            line = raw_line.strip()
            # 跳过空行和注释行，允许 .env 中写说明文字。
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


load_env_file()

# DashScope 兼容 OpenAI 接口的访问密钥。
# 优先读取环境变量，真实密钥请写入本地 .env 文件，不要提交到 Git。
API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not API_KEY:
    raise RuntimeError("缺少 DASHSCOPE_API_KEY，请在本地 .env 文件或系统环境变量中配置。")

MODEL_NAME = "qwen3-235b-a22b-instruct-2507"

# 初始化客户端
client = OpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=API_KEY,
)

# 实体类型标签映射
TYPE_TAGS = {
    "性格特质": "personality",
    "品德行为": "moral",
    "学习表现": "study",
    "不足之处": "weakness",
    "期望建议": "suggestion"
}

# 反向映射
TAG_TO_TYPE = {v: k for k, v in TYPE_TAGS.items()}


SYSTEM_PROMPT = """
<identity>
你是一个信息抽取助手，对于中文文本，你有着深刻的理解，能够精准捕捉语义之间的联系，高质量的完成用户的要求。
</identity>

<purpose>
你需要对用户提供的语料（可能有多条）按照rules_and_examples中的规则判断语料中的实体以及实体类型。
请务必不要直接输出结果，你应该按照先输出<think>标签，并在think标签中输出你具体的思考过程，不用担心上下文长度的问题，所以你可以尽情的思考发挥。一旦思考完成，你需要输出一个<result>标签，并在result标签中严格按output_format中规定的格式进行输出结果。请务必遵守这条原则，以免带来不可控的错误。
</purpose>

<rules_and_examples>
# 一、标注规则

## 1.实体定义

性格特质：描述学生稳定性格和心理特征的词汇

品德行为：涉及道德规范和人际交往的行为描述

学习表现：学习态度、能力和成果的正向表述

不足之处：需改进的缺点或薄弱环节

期望建议：教师提出的改进方向或目标

 

## 2.边界处理

剔除指代词"你、孩子、同学、他、她"后确定实体边界

起始位置为实体描述第一个字，结束位置为最后一个字

忽略"老师觉得、我发现"等引导语

 

## 3.拆分

同类型或不同类型紧邻描述必须拆分（如"认真听讲但偶尔走神"→拆分为2个实体）

转折词（但、却、然而）连接的内容强制拆分



## 4.输出格式

```json
{
    "labeled_texts": [
        "labeled_text1",
        "labeled_text2",
        ...
    ]
}
```

# 二、实体的核心判别标准

1. 有明确指向性：描述的是学生的具体特征、行为、表现或教师的具体期望（而非泛泛而谈的套话）。

例：“乐于助人”（指向具体品德）是实体；“继续努力”（无具体指向）非实体。

2. 有实际意义：能独立反映学生的某一属性（性格、品德、学习等）或教师的具体建议。

例：“作业马虎”（反映学习中的不足）是实体；“加油哦”（无实际意义）非实体。

3. 非指代词/辅助词：剔除“你、孩子、同学”等指代词和“老师觉得、希望你”等引导语后，剩余部分仍有完整意义。

例：“你上课认真听讲”→“上课认真听讲”（学习表现）是实体。

 

# 三、实体判别特征

| 实体类型 | 核心特征                                                     | 标志词                                                       | 排除                                                         |
| -------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 性格特质 | 描述学生稳定的性格、心理倾向（无明确行为或学习指向，侧重“内在属性”）。 | 向，侧重“内在属性”）。 开朗、内向、细心、急躁、自信、胆怯、活泼、文静、倔强、诚实、善良等形容词。 | 临时情绪（如“今天很开心”）、因场景变化的表现（如“考试时紧张”）。 |
| 品德行为 | 描述学生在人际交往、集体生活中的道德行为或规范遵守（侧重“对外行为”）。 | 尊敬师长、团结同学、乐于助人、遵守纪律、热爱集体、爱护公物、讲卫生、懂礼貌等。 | 与道德/规范无关的行为（如“喜欢打篮球”）。                    |
| 学习表现 | 描述学生在学习态度、能力、成果上的正向表现（侧重“学业相关的积极面”）。 | 勤奋刻苦、思维敏捷、成绩优异、善于思考、认真听讲、作业工整、求知欲强等。 | 中性的学习状态（如“按时上课”）、学习中的不足（如“成绩落后”）。 |
| 不足之处 | 描述学生需改进的缺点、薄弱环节（侧重“负面或待提升的方面”）。 | 粗心马虎、注意力不集中、作业拖沓、上课走神、贪玩、书写潦草、偏科等。 | 客观事实陈述（如“身高较矮”）                                 |
| 期望建议 | 教师对学生提出的具体改进方向、目标或方法（侧重“祈使句、建议性内容”）。 | 希…、希望…、建议…、要…、要是…、需…、需要…、应…、争取…、望…等引导的句子。 | 泛泛的鼓励（如“继续加油”）、无具体措施的期望（如“希望你更好”）。 |

判断流程：

**第一步：过滤非实体内容**

忽略指代词：“你、孩子、同学、他、她”等（保留核心描述）。

忽略辅助语：“老师觉得、我发现、总的来说”等引导语（保留主体内容）。

**第二步：拆分语义单元**

按标点（逗号、分号、句号）或转折词（但、却、然而）拆分句子为独立语义单元。

（例：“认真听讲但偶尔走神”→拆分为“认真听讲”“偶尔走神”两个单元。）

**第三步：判断是否为实体**

对每个语义单元，用“核心判别标准”检验：是否有明确指向性、实际意义、非辅助词。

（例：“认真听讲”→指向学习态度，有实际意义→是实体；“哦”→无实际意义→非实体。）

**第四步：匹配实体类型**

根据“五大类实体特征”，给每个实体匹配类型（优先按“负面/建议”→“品德/学习”→“性格”的优先级判断，避免混淆）。

（例：“粗心马虎”→负面且待改进→匹配“不足之处”；“乐于助人”→道德行为→匹配“品德行为”。）



 判断偏好：

不足之处（负面描述优先）、期望建议（祈使句优先）、品德行为（人际相关优先）、学习表现（学业相关优先）、性格特质（剩余的性格描述）

 

易混淆实体的判断优先级：

1. 负面优先：含负面描述的短语（如粗心、走神）优先判定为“不足之处”，排除学习表现/性格特质。

2. 建议优先：含祈使句（希望、应、要）的短语优先判定为“期望建议”，排除其他类型。

3. 行为优先：涉及人际交往/集体规范的行为（如助人、守纪）优先判定为“品德行为”，排除学习表现。

4. 学业优先：指向学习态度/能力的表述（如认真听讲、解题快）优先判定为“学习表现”，排除性格特质。

5. 剩余归性格：无法归入前四类的稳定特质（如聪明、文静），判定为“性格特质”。

# 四、示例判定

## 4.1 口语化文本

### 1. 类型判断

| 口语化表述             | 蕴含含义                 | 实体类型 |
| ---------------------- | ------------------------ | -------- |
| 焉儿坏，爱开玩笑       | 内向但调皮，幽默风趣     | 性格特质 |
| 虎了吧唧，天不怕地不怕 | 鲁莽，勇敢无畏           | 性格特质 |
| 眼里有活，爱帮人       | 热爱劳动，乐于助人       | 品德行为 |
| 护犊子，维护班级荣誉   | 有集体荣誉感             | 品德行为 |
| 脑子转得快，一点就透   | 思维敏捷，理解力强       | 学习表现 |
| 难题啃得下来           | 善于钻研，解决问题能力强 | 学习表现 |
| 嘴碎，上课爱搭话       | 爱讲闲话，违反纪律       | 不足之处 |
| 坐不住，屁股上长钉子   | 注意力不集中，好动       | 不足之处 |
| 不会就问，别憋在心里   | 建议主动请教             | 期望建议 |
| 字写工整点，别糊弄     | 建议书写规范，认真对待   | 期望建议 |

### 2. 实体拆分

| 口语化表述               | 拆分实体及类型                               |
| ------------------------ | -------------------------------------------- |
| 聪明是聪明，就是不爱学   | 聪明（性格特质）；不爱学（不足之处）         |
| 干活勤快，做题马虎       | 干活勤快（品德行为）；做题马虎（不足之处）   |
| 脑子灵，就是不用在正地方 | 脑子灵（性格特质）；不用在正地方（不足之处） |
|                          |                                              |

## 4.2 间接表达

间接表达因省略、隐喻、模糊情感等特点，容易导致实体边界模糊或类型误判。

| 示例                               | 隐含实体类型           |
| ---------------------------------- | ---------------------- |
| 你的笔锋像弹簧，时紧时松           | 不足之处（书写不认真） |
| 课堂上的你像害羞的含羞草，总低着头 | 性格特质（内向胆怯）   |
| 知识的海洋里，你像断了桨的船       | 学习表现（方法不当）   |
| 你其实可以做的更好，只是还没找到钥匙 | 期望建议（希望改进学习方法） |
| 这次没发挥好，下次别让老师失望       | 不足之处（发挥失常） |
| 这次进步的背后，藏着不少深夜的灯光 | 学习表现（勤奋刻苦） |
| 小组讨论时，你的声音总被淹没 | 性格特质（内向） |
| 要是再专注点就好了 | 不足之处（注意力不集中） |
| 那道题其实你会的 | 学习表现（有潜力但未发挥） |

参考判别方法：

1. 语义补全法：
   对省略式表达，补充主语（学生）和场景（学习/人际/性格），如“要是再专注点就好了”→“（你）在学习上要是再专注点就好了”，明确指向“不足之处”。

2. 隐喻破解法：
   对比喻句，提取本体和喻体的关联，如“笔锋像弹簧”→“书写松紧不一”→“书写不认真”（不足之处）。

3. 因果推导法：
   对结果描述，推导隐含的原因，如“红叉比星星多”→“错误多”→“粗心”（不足之处）。

4. 情感倾向判断法：
   对模糊评价，通过关联词（“只是”“其实”“要是”）判断褒贬，如“只是还没找到钥匙”中“只是”暗示存在不足，指向“期望建议”。

</rules_and_examples>

<output_format>
使用xml格式对所给语句进行标注，每个实体用对应类型标签包裹，实体有且仅有以下5种：
- 性格特质：personality
- 品德行为：moral
- 学习表现：study
- 不足之处：weakness
- 期望建议：suggestion

**IMPORTANT**：确保输出的result中，除了实体标签外文字与所给语句完全一致。

请输出think过程，并将最后的结果按照以下格式进行输出（不包括三反引号）：
```json
{
    "result": "无关语句<personality>实体1</personality>，<study>实体2</study>，无关语句<weakness>实体3</weakness>。无关语句，<suggestion>实体4</suggestion>，无关语句。"
}
```
</output_format>

<output_example>
原文：孩子活泼开朗，上课认真听讲，偶尔会讲小话，希望能改进，加油。

输出：
<think>
思考过程...
</think>
<result>
```json
{
    "result": "孩子<personality>活泼开朗</personality>，<study>上课认真听讲</study>，偶尔会<weakness>讲小话</weakness>，<suggestion>希望能改进</suggestion>，加油。"
}
```
</result>
</output_example>
""".strip()


def call_doubao_api(prompt: str, status_prefix: str = "") -> str:
    """调用API并返回标注文本"""
    retry_count = 3
    for i in range(retry_count):
        try:
            with Status(f"{status_prefix} 正在提取...") as status:
                reply = ""
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=8192,
                    temperature=0.1,
                    top_p=0.5,
                    stream=True
                )
                for chunk in completion:
                    if chunk.choices[0].delta.content:
                        reply += chunk.choices[0].delta.content
                        status.update(f"{status_prefix} 正在提取中: {json.dumps(reply[-50:], ensure_ascii=False)}...")
                return reply.strip()
        except Exception as e:
            print(f"第{i + 1}次调用失败: {e}")
            if i == retry_count - 1:
                raise

def load_comments(file_path: str) -> List[str]:
    """读取评语数据集并过滤重复文本"""
    with open(file_path, "r", encoding="utf-8") as f:
        comments = [line.strip() for line in f if line.strip()]
    unique_comments = list(dict.fromkeys(comments))
    print(f"原始评语数: {len(comments)}, 去重后: {len(unique_comments)}")
    return unique_comments

def parse_tagged_text(tagged_text: str, original_text: str) -> Dict:
    """解析标签文本，转换为JSON格式"""
    # 从 tagged_text 中提取 ```json 包裹的内容
    result = re.search(r'```json\s*(.*?)\s*```', tagged_text, re.DOTALL).group(1).strip()
    result = json.loads(result)
    
    tagged_text = result["result"]
    
    entities = []
    
    # 查找所有标签
    pattern = r'<(personality|moral|study|weakness|suggestion)>(.*?)</\1>'
    matches = re.finditer(pattern, tagged_text)
    
    for match in matches:
        tag_type = match.group(1)
        entity_text = match.group(2).strip()
        
        if not entity_text:
            continue
            
        # 在原文中查找实体位置
        entity_start = original_text.find(entity_text)
        if entity_start == -1:
            # 如果找不到完全匹配，尝试模糊匹配
            for i in range(len(original_text) - len(entity_text) + 1):
                if original_text[i:i+len(entity_text)].replace(" ", "") == entity_text.replace(" ", ""):
                    entity_start = i
                    break
        
        if entity_start != -1:
            entity = {
                "label_type": TAG_TO_TYPE[tag_type],
                "overlap": 0,
                "start_pos": entity_start,
                "end_pos": entity_start + len(entity_text)
            }
            entities.append(entity)
    
    # 按位置排序
    entities.sort(key=lambda x: x["start_pos"])
    
    # 去重和处理重叠
    final_entities = []
    for entity in entities:
        # 检查是否与已有实体重叠
        overlap = False
        for existing in final_entities:
            if (entity["start_pos"] < existing["end_pos"] and 
                entity["end_pos"] > existing["start_pos"]):
                overlap = True
                break
        
        if not overlap:
            final_entities.append(entity)
    
    return {
        "originalText": original_text,
        "labeled_texts": tagged_text,
        "entities": final_entities,
    }

def process_annotation(comment: str, status_prefix: str = "") -> Dict:
    """处理单条评语的标注"""
    try:
        # 1. 调用API获取标注文本
        tagged_text = call_doubao_api(comment, status_prefix=status_prefix)
        # print(f"标注结果: {tagged_text}")
        
        # 2. 解析标签文本为JSON格式
        result = parse_tagged_text(tagged_text, comment)
        # print(f"解析结果: {json.dumps(result, ensure_ascii=False, indent=2)}")
        
        return result
    except Exception as e:
        print(f"处理失败: {str(e)}")
        return {"originalText": comment, "entities": []}

def batch_process_and_save(comments: List[str], output_file: str) -> None:
    """批量处理并保存结果"""
    processed = {}
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if data["labeled_texts"].strip() == "":
                        continue  # 跳过空的标注文本
                    if "entities" not in data or not data["entities"]:
                        continue  # 跳过没有实体的结果
                    processed[data["originalText"]] = data
                except:
                    continue

    print(f"已处理 {len(processed)} 条，剩余 {len(comments) - len(processed)} 条")
    results = list(processed.values())
    new_count = 0

    for i, comment in enumerate(comments):
        if comment in processed:
            continue

        result = process_annotation(comment, status_prefix=f"[{i + 1}/{len(comments)}]")
        results.insert(i, result)
        new_count += 1

        # 每20条保存一次
        if new_count % 20 == 0 or (i + 1) == len(comments):
            with open(output_file, "w", encoding="utf-8") as f:
                for item in results:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"已保存 {len(results)} 条结果")
    # 最后保存一次
    with open(output_file, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    if new_count > 0:
        print(f"新增 {new_count} 条标注结果")
    else:
        print("没有新增标注结果")

    print(f"全部完成，共处理 {len(results)} 条")


if __name__ == "__main__":
    input_comments_file = "comments/comments_add.txt"
    final_output_file = "comments/comments_labeled_add.jsonl"
    os.makedirs(os.path.dirname(input_comments_file), exist_ok=True)
    os.makedirs(os.path.dirname(final_output_file), exist_ok=True)
    comments = load_comments(input_comments_file)  # 限制处理前10条评论以测试
    batch_process_and_save(comments, final_output_file)
