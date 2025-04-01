# **📌 代码解析：GLUE 任务的 PyTorch Lightning 训练脚本**

这个代码用于 **在 GLUE 任务上微调 Transformer 模型**，使用 **PyTorch Lightning 进行封装**，并 **自动管理训练、验证、测试等流程**。其中包含：
- **支持 GLUE 多分类任务（文本分类）**
- **支持 BERT、XLNet、RoBERTa、ALBERT**
- **使用 `transformers` 进行数据处理**
- **基于 `pytorch_lightning` 进行封装**
- **自动管理 `logging` 和 `checkpoint`**

---

## **📌 1. 代码主要功能**
✅ **基于 `PyTorch Lightning` 进行 Transformer 微调**  
✅ **支持 `BERT`、`XLNet`、`RoBERTa`、`ALBERT` 等预训练模型**  
✅ **支持 `GLUE` 任务，如 `SST-2`（情感分类）、`MRPC`（句子相似度）**  
✅ **自动加载 & 处理数据**
✅ **自动 `checkpoint` & `logging`**  
✅ **支持 `GPU` 训练**

---

# **📌 2. 代码解析**
## **1️⃣ `GLUETransformer`（核心模型）**
```python
class GLUETransformer(BaseTransformer):
    mode = "sequence-classification"

    def __init__(self, hparams):
        if isinstance(hparams, dict):
            hparams = Namespace(**hparams)
        hparams.glue_output_mode = glue_output_modes[hparams.task]
        num_labels = glue_tasks_num_labels[hparams.task]

        super().__init__(hparams, num_labels, self.mode)
```
📌 **作用**：
- **继承 `BaseTransformer`，用于 `GLUE` 任务的微调**
- **根据 `GLUE` 任务类别，自动设置 `num_labels`**
- **自动配置 `glue_output_mode`（分类或回归）**

---

### **2️⃣ `training_step()`（训练步骤）**
```python
def training_step(self, batch, batch_idx):
    inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
    if self.config.model_type not in ["distilbert", "bart"]:
        inputs["token_type_ids"] = batch[2] if self.config.model_type in ["bert", "xlnet", "albert"] else None

    outputs = self(**inputs)
    loss = outputs[0]

    lr_scheduler = self.trainer.lr_schedulers[0]["scheduler"]
    tensorboard_logs = {"loss": loss, "rate": lr_scheduler.get_last_lr()[-1]}
    return {"loss": loss, "log": tensorboard_logs}
```
📌 **作用**：
- **执行前向传播**
- **计算 `loss`**
- **记录 `loss` & `learning_rate`**
- **支持不同模型（BERT、XLNet、ALBERT）**

---

### **3️⃣ `prepare_data()`（数据预处理）**
```python
def prepare_data(self):
    args = self.hparams
    processor = processors[args.task]()
    self.labels = processor.get_labels()

    for mode in ["train", "dev"]:
        cached_features_file = self._feature_file(mode)
        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
        else:
            logger.info("Creating features from dataset file at %s", args.data_dir)
            examples = (
                processor.get_dev_examples(args.data_dir)
                if mode == "dev"
                else processor.get_train_examples(args.data_dir)
            )
            features = convert_examples_to_features(
                examples,
                self.tokenizer,
                max_length=args.max_seq_length,
                label_list=self.labels,
                output_mode=args.glue_output_mode,
            )
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)
```
📌 **作用**：
- **自动处理 `GLUE` 任务数据**
- **使用 `convert_examples_to_features()` 进行数据转换**
- **自动缓存数据，提高加载效率**

---

### **4️⃣ `get_dataloader()`（数据加载）**
```python
def get_dataloader(self, mode: str, batch_size: int, shuffle: bool = False) -> DataLoader:
    mode = "dev" if mode == "test" else mode
    cached_features_file = self._feature_file(mode)
    logger.info("Loading features from cached file %s", cached_features_file)
    features = torch.load(cached_features_file)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if self.hparams.glue_output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif self.hparams.glue_output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    return DataLoader(
        TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels),
        batch_size=batch_size,
        shuffle=shuffle,
    )
```
📌 **作用**：
- **自动加载数据**
- **转换为 `TensorDataset`**
- **返回 `DataLoader`**

---

### **5️⃣ `validation_step()`（验证步骤）**
```python
def validation_step(self, batch, batch_idx):
    inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
    if self.config.model_type not in ["distilbert", "bart"]:
        inputs["token_type_ids"] = batch[2] if self.config.model_type in ["bert", "xlnet", "albert"] else None

    outputs = self(**inputs)
    tmp_eval_loss, logits = outputs[:2]
    preds = logits.detach().cpu().numpy()
    out_label_ids = inputs["labels"].detach().cpu().numpy()

    return {"val_loss": tmp_eval_loss.detach().cpu(), "pred": preds, "target": out_label_ids}
```
📌 **作用**：
- **前向传播**
- **计算 `loss`**
- **返回预测值 & 真实标签**

---

### **6️⃣ `validation_epoch_end()`（验证结束）**
```python
def validation_epoch_end(self, outputs: list) -> dict:
    ret, preds, targets = self._eval_end(outputs)
    logs = ret["log"]
    return {"val_loss": logs["val_loss"], "log": logs, "progress_bar": logs}
```
📌 **作用**：
- **计算最终 `val_loss`**
- **记录日志**

---

### **7️⃣ `test_epoch_end()`（测试步骤）**
```python
def test_epoch_end(self, outputs) -> dict:
    ret, predictions, targets = self._eval_end(outputs)
    logs = ret["log"]
    return {"avg_test_loss": logs["val_loss"], "log": logs, "progress_bar": logs}
```
📌 **作用**：
- **计算最终 `test_loss`**
- **记录日志**

---

### **8️⃣ `main()`（训练入口）**
```python
def main():
    parser = argparse.ArgumentParser()
    add_generic_args(parser, os.getcwd())
    parser = GLUETransformer.add_model_specific_args(parser, os.getcwd())
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(
            "./results",
            f"{args.task}_{time.strftime('%Y%m%d_%H%M%S')}",
        )
        os.makedirs(args.output_dir)

    model = GLUETransformer(args)
    trainer = generic_train(model, args)

    if args.do_predict:
        checkpoints = sorted(glob.glob(os.path.join(args.output_dir, "checkpoint-epoch=*.ckpt"), recursive=True))
        model = model.load_from_checkpoint(checkpoints[-1])
        return trainer.test(model)
```
📌 **作用**：
- **解析参数**
- **初始化 `GLUETransformer`**
- **执行 `trainer.fit()` 进行训练**
- **支持 `checkpoint` 加载**

---

# **📌 3. 代码总结**
✅ **基于 `pytorch_lightning` 进行 `GLUE` 任务微调**  
✅ **支持 `BERT`、`XLNet`、`RoBERTa`**  
✅ **自动 `checkpoint` & `logging`**  
✅ **支持 `GPU` 训练**  

🚀 **适用于 `多 GPU 训练`，提高训练效率！**