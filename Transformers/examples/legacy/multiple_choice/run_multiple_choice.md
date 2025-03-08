这段代码用于 **微调（Finetune）Hugging Face Transformers 库中的多项选择（Multiple Choice）模型**，支持 **BERT、RoBERTa、XLNet** 这类预训练模型。

---

## **📌 1. 代码主要功能**
✅ **支持微调多项选择（Multiple Choice）任务（如 SWAG, ARC, RACE）**  
✅ **加载 Hugging Face 预训练模型并适配任务**  
✅ **支持 `Trainer` API 进行训练和评估**  
✅ **可选择 GPU/TPU 进行加速**  
✅ **自动计算准确率（Accuracy）**

---

## **📌 2. 代码解析**
### **1️⃣ 解析命令行参数**
```python
parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()
```
- **`ModelArguments`**：模型相关参数（模型名称、缓存目录等）
- **`DataTrainingArguments`**：数据相关参数（数据目录、最大序列长度等）
- **`TrainingArguments`**：训练相关参数（批次大小、学习率、训练步数等）

✅ **示例**
```bash
python run_multiple_choice.py \
    --model_name_or_path bert-base-uncased \
    --task_name swag \
    --data_dir ./data \
    --output_dir ./output \
    --do_train \
    --do_eval
```

---

### **2️⃣ 设定日志**
```python
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
)
```
- **`local_rank in [-1, 0]`**：只在 **主进程** 记录日志（避免多 GPU 并行训练时重复打印）

---

### **3️⃣ 加载数据集**
```python
processor = processors[data_args.task_name]()
label_list = processor.get_labels()
num_labels = len(label_list)
```
- **从 `processors` 加载指定任务的处理器**
- **获取所有类别的标签**
- **计算 `num_labels` 用于配置模型**

---
### **4️⃣ 加载预训练模型**
```python
config = AutoConfig.from_pretrained(
    model_args.config_name if model_args.config_name else model_args.model_name_or_path,
    num_labels=num_labels,
    finetuning_task=data_args.task_name,
)
tokenizer = AutoTokenizer.from_pretrained(
    model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
)
model = AutoModelForMultipleChoice.from_pretrained(
    model_args.model_name_or_path,
    config=config,
)
```
- **`AutoConfig`**：加载模型配置（自动匹配 `num_labels`）
- **`AutoTokenizer`**：加载分词器（自动匹配 `BERT / RoBERTa / XLNet`）
- **`AutoModelForMultipleChoice`**：加载预训练模型

✅ **支持的模型**
- `bert-base-uncased`
- `roberta-base`
- `xlnet-base-cased`
- `albert-base-v2`
- `deberta-v3-base`
- `gpt2`（部分多选任务适配）

---

### **5️⃣ 训练集 & 验证集**
```python
train_dataset = (
    MultipleChoiceDataset(
        data_dir=data_args.data_dir,
        tokenizer=tokenizer,
        task=data_args.task_name,
        max_seq_length=data_args.max_seq_length,
        overwrite_cache=data_args.overwrite_cache,
        mode=Split.train,
    )
    if training_args.do_train
    else None
)
eval_dataset = (
    MultipleChoiceDataset(
        data_dir=data_args.data_dir,
        tokenizer=tokenizer,
        task=data_args.task_name,
        max_seq_length=data_args.max_seq_length,
        overwrite_cache=data_args.overwrite_cache,
        mode=Split.dev,
    )
    if training_args.do_eval
    else None
)
```
🔹 **`MultipleChoiceDataset`**：对 **输入文本对（question, choice）进行 Tokenization 处理**
🔹 **`mode=Split.train/dev`**：支持 **训练集（train）和验证集（dev）**

✅ **示例**
- **SWAG 任务（句子预测）**
  ```python
  context = "She went to the kitchen."
  choices = ["She grabbed an apple.", "He played basketball."]
  ```
  - 目标是从 `choices` 中 **选择最合理的句子**
---

### **6️⃣ 计算准确率**
```python
def compute_metrics(p: EvalPrediction) -> Dict:
    preds = np.argmax(p.predictions, axis=1)
    return {"acc": simple_accuracy(preds, p.label_ids)}
```
- **`p.predictions`**：模型输出的 `logits`
- **`np.argmax(..., axis=1)`**：选择概率最高的选项
- **`simple_accuracy`**：计算准确率

---

### **7️⃣ 训练 & 评估**
```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
)

if training_args.do_train:
    trainer.train()
    trainer.save_model()
```
- **`Trainer`** 统一管理训练流程（支持 GPU/TPU）
- **`trainer.train()`** 进行模型训练
- **`trainer.save_model()`** 保存模型

✅ **示例**
```bash
python run_multiple_choice.py --do_train --do_eval
```

---

## **📌 3. 训练命令**
**✅ SWAG 任务**
```bash
python run_multiple_choice.py \
    --model_name_or_path bert-base-uncased \
    --task_name swag \
    --data_dir ./data \
    --output_dir ./output \
    --do_train \
    --do_eval \
    --max_seq_length 128 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --learning_rate 2e-5 \
    --num_train_epochs 3
```

---

## **📌 4. 代码总结**
✅ **适用于多项选择任务（Multiple Choice）**  
✅ **支持 `BERT`、`RoBERTa`、`XLNet` 预训练模型**  
✅ **自动 Tokenize 任务数据，支持 `Trainer` 训练**  
✅ **自动计算准确率（Accuracy）**  
✅ **可用于多个 NLP 任务（如 SWAG, RACE, ARC）**

🚀 **可以尝试不同的 `batch_size / learning_rate` 找到最佳超参数！**