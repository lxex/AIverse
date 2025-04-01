这段代码是 **多项选择（Multiple Choice）任务的工具代码**，用于 **处理数据集（SWAG、RACE、ARC、Synonym）**，并将数据转换为 **适用于 Transformer 模型的格式**。

---

## **📌 1. 代码主要功能**
✅ **支持多个多项选择任务（SWAG、RACE、ARC、Synonym）**  
✅ **加载数据集并转换为 `InputExample` 格式**  
✅ **支持 PyTorch & TensorFlow 训练数据格式**  
✅ **Tokenize 任务文本，转换为 `InputFeatures` 适配 Transformer**  
✅ **缓存处理后的数据，加速训练**  
✅ **数据集格式适配 BERT、RoBERTa、XLNet 等 Transformer 模型**  

---

## **📌 2. 代码解析**
### **1️⃣ 关键数据结构**
#### **① `InputExample`（表示一个训练样本）**
```python
@dataclass(frozen=True)
class InputExample:
    example_id: str
    question: str
    contexts: List[str]  # 选项对应的上下文
    endings: List[str]   # 多项选择的选项
    label: Optional[str] # 选项的正确答案
```
✅ **示例（SWAG）**
```python
InputExample(
    example_id="001",
    question="She went to the kitchen.",
    contexts=["She grabbed an apple.", "He played basketball."],
    endings=["She ate it.", "He threw it away."],
    label="0"  # 代表第一个选项正确
)
```
---

#### **② `InputFeatures`（模型输入格式）**
```python
@dataclass(frozen=True)
class InputFeatures:
    example_id: str
    input_ids: List[List[int]]
    attention_mask: Optional[List[List[int]]]
    token_type_ids: Optional[List[List[int]]]
    label: Optional[int]
```
✅ **转换示例**
- `input_ids`：模型输入的 token ID
- `attention_mask`：用于填充的 mask
- `token_type_ids`：区分 `context` 和 `question`
- `label`：正确选项索引（如 `0`）

---

### **2️⃣ 任务数据集处理**
支持 **PyTorch 和 TensorFlow** 两种格式  
✅ **PyTorch 格式**
```python
class MultipleChoiceDataset(Dataset):
    def __init__(self, data_dir: str, tokenizer: PreTrainedTokenizer, task: str, ...):
        processor = processors[task]()
        cached_features_file = os.path.join(data_dir, "cached_{}_{}_{}".format(...))
        with FileLock(cached_features_file + ".lock"):
            if os.path.exists(cached_features_file):
                self.features = torch.load(cached_features_file)
            else:
                examples = processor.get_train_examples(data_dir)
                self.features = convert_examples_to_features(examples, tokenizer, max_seq_length)
                torch.save(self.features, cached_features_file)
```
📌 **作用**：
- 解析任务数据（RACE, SWAG, ARC, Synonym）
- 使用 **Tokenizer 进行 tokenization**
- **缓存处理后数据**，避免重复处理，加速训练

✅ **TensorFlow 格式**
```python
class TFMultipleChoiceDataset:
    def __init__(self, data_dir: str, tokenizer: PreTrainedTokenizer, task: str, ...):
        processor = processors[task]()
        examples = processor.get_train_examples(data_dir)
        self.features = convert_examples_to_features(examples, tokenizer, max_seq_length)
```
📌 **作用**：
- 处理数据格式适配 TensorFlow
- 使用 `tf.data.Dataset` 进行 **数据流式加载**

---

### **3️⃣ 任务数据处理器**
每个任务有不同的数据格式，代码提供了不同的 `Processor`：

#### **① RACE 任务（阅读理解）**
```python
class RaceProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        high = self._read_txt(os.path.join(data_dir, "train/high"))
        middle = self._read_txt(os.path.join(data_dir, "train/middle"))
        return self._create_examples(high + middle, "train")

    def _read_txt(self, input_dir):
        lines = []
        for file in glob.glob(input_dir + "/*txt"):
            with open(file, "r", encoding="utf-8") as fin:
                data_raw = json.load(fin)
                data_raw["race_id"] = file
                lines.append(data_raw)
        return lines
```
✅ **作用**
- 读取 RACE 数据集 **（分为 high/middle 级别）**
- 解析 JSON 数据，抽取 **`question`、`contexts` 和 `endings`**
- 生成 `InputExample`

---

#### **② SWAG 任务（故事填空）**
```python
class SwagProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_csv(os.path.join(data_dir, "train.csv")), "train")

    def _read_csv(self, input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            return list(csv.reader(f))

    def _create_examples(self, lines: List[List[str]], type: str):
        return [
            InputExample(
                example_id=line[2],
                question=line[5],
                contexts=[line[4], line[4], line[4], line[4]],
                endings=[line[7], line[8], line[9], line[10]],
                label=line[11],
            )
            for line in lines[1:]
        ]
```
✅ **作用**
- 读取 SWAG 数据集（CSV 格式）
- `question` 是上下文的补充部分
- 解析 **四个选项（`endings`）**
- 生成 `InputExample`

---

#### **③ ARC 任务（科学选择题）**
```python
class ArcProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_json(os.path.join(data_dir, "train.jsonl")), "train")

    def _read_json(self, input_file):
        with open(input_file, "r", encoding="utf-8") as fin:
            return [json.loads(line.strip("\n")) for line in fin.readlines()]

    def _create_examples(self, lines, type):
        examples = []
        for data_raw in lines:
            question = data_raw["question"]["stem"]
            options = data_raw["question"]["choices"]
            label = str(ord(data_raw["answerKey"]) - ord("A"))
            examples.append(
                InputExample(
                    example_id=data_raw["id"],
                    question=question,
                    contexts=[o["para"].replace("_", "") for o in options],
                    endings=[o["text"] for o in options],
                    label=label,
                )
            )
        return examples
```
✅ **作用**
- 读取 ARC 数据集（JSONL 格式）
- 处理 **科学选择题**
- 解析 **`question`、`contexts`、`endings`**
- 生成 `InputExample`

---

### **4️⃣ 数据转换为模型输入**
```python
def convert_examples_to_features(examples, label_list, max_length, tokenizer):
    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for example in examples:
        choices_inputs = []
        for context, ending in zip(example.contexts, example.endings):
            inputs = tokenizer(
                context, ending, add_special_tokens=True, max_length=max_length,
                padding="max_length", truncation=True
            )
            choices_inputs.append(inputs)

        input_ids = [x["input_ids"] for x in choices_inputs]
        attention_mask = [x["attention_mask"] for x in choices_inputs]
        token_type_ids = [x["token_type_ids"] for x in choices_inputs]

        features.append(
            InputFeatures(
                example_id=example.example_id,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                label=label_map[example.label],
            )
        )
    return features
```
📌 **作用**
- **调用 `tokenizer` 对文本进行 Tokenization**
- **转换为 `input_ids`、`attention_mask`、`token_type_ids`**
- **转换 `label` 为整数索引**

---

## **📌 3. 代码总结**
✅ **支持 SWAG、RACE、ARC 任务**  
✅ **自动处理数据格式，并进行 Tokenization**  
✅ **适配 Transformer 预训练模型**  
✅ **自动缓存处理后数据，加速训练**  

🚀 **适用于 `BERT`、`RoBERTa`、`XLNet` 进行多项选择任务训练！**