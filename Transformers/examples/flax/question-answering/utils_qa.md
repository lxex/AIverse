这段代码实现了 **问答（QA）模型的后处理**，它从 **模型预测的 `start_logits` 和 `end_logits` 计算出最终答案**。该代码主要适用于 **SQuAD 格式** 的问答数据集。

---

## **1. 代码结构**
```
|-- postprocess_qa_predictions()           # 适用于一般模型（如 BERT, RoBERTa）
|-- postprocess_qa_predictions_with_beam_search()  # 适用于使用 beam search 预测的模型（如 XLNet, ALBERT）
```
两者的主要区别：
- `postprocess_qa_predictions()` 处理 **一般情况下的 start & end logits**。
- `postprocess_qa_predictions_with_beam_search()` 处理 **带有 beam search 的预测**（例如 `XLNet`，它会返回多个 `top-k` 预测）。

---

## **2. 代码解析**
### **(1) `postprocess_qa_predictions()` 主要流程**
```python
def postprocess_qa_predictions(
    examples,
    features,
    predictions: Tuple[np.ndarray, np.ndarray],  # (start_logits, end_logits)
    version_2_with_negative: bool = False,
    n_best_size: int = 20,
    max_answer_length: int = 30,
    null_score_diff_threshold: float = 0.0,
    output_dir: Optional[str] = None,
    prefix: Optional[str] = None,
    log_level: Optional[int] = logging.WARNING,
):
```
- `examples`: **原始未处理数据**（如 `SQuAD` 数据）。
- `features`: **经过 `tokenizer` 处理的数据**，包含 `offset_mapping`（字符偏移信息）。
- `predictions`: **模型输出的 `(start_logits, end_logits)`**。
- `version_2_with_negative`: **是否支持 `SQuAD v2` 的 "无法回答" 选项**。
- `n_best_size`: **保留 `n_best` 个候选答案**。
- `max_answer_length`: **答案最大长度**，防止输出过长答案。
- `null_score_diff_threshold`: **选择 "无答案" 的阈值**（用于 `SQuAD v2`）。
- `output_dir`: **如果指定，存储 `JSON` 结果**。

---

### **(2) 构建 `example_id` 与 `features` 的映射**
```python
example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
features_per_example = collections.defaultdict(list)
for i, feature in enumerate(features):
    features_per_example[example_id_to_index[feature["example_id"]]].append(i)
```
- **`example_id_to_index`**: `SQuAD` 数据中的每个 `id` 与索引的映射。
- **`features_per_example`**: 由于一个 `context` 可能被 `tokenizer` 切分成多个 `feature`，所以需要存储一个 `example_id` 对应的 `feature`。

---

### **(3) 遍历所有样本**
```python
for example_index, example in enumerate(tqdm(examples)):
    feature_indices = features_per_example[example_index]
    min_null_prediction = None
    prelim_predictions = []
```
- **遍历 `SQuAD` 样本**。
- **获取当前样本对应的所有 `features`**。
- **`min_null_prediction` 用于存储 "无答案" 情况的最低分数**。
- **`prelim_predictions` 用于存储所有可能的答案**。

---

### **(4) 遍历所有 `features` 并计算 `n_best`**
```python
for feature_index in feature_indices:
    start_logits = all_start_logits[feature_index]
    end_logits = all_end_logits[feature_index]
    offset_mapping = features[feature_index]["offset_mapping"]
    token_is_max_context = features[feature_index].get("token_is_max_context", None)

    feature_null_score = start_logits[0] + end_logits[0]
    if min_null_prediction is None or min_null_prediction["score"] > feature_null_score:
        min_null_prediction = {
            "offsets": (0, 0),
            "score": feature_null_score,
            "start_logit": start_logits[0],
            "end_logit": end_logits[0],
        }

    # 获取 `start_logits` 和 `end_logits` 中 `n_best_size` 个最大索引
    start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
    end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
```
- **计算 `start_logits` 和 `end_logits` 的 `n_best_size` 个索引**。
- **`offset_mapping`** 存储的是 `context` 中每个 `token` 对应的字符位置。

---

### **(5) 计算 `n_best` 答案**
```python
for start_index in start_indexes:
    for end_index in end_indexes:
        # 筛选逻辑：索引不能越界、长度不能超过 `max_answer_length`
        if (
            start_index >= len(offset_mapping)
            or end_index >= len(offset_mapping)
            or end_index < start_index
            or end_index - start_index + 1 > max_answer_length
        ):
            continue
        prelim_predictions.append(
            {
                "offsets": (offset_mapping[start_index][0], offset_mapping[end_index][1]),
                "score": start_logits[start_index] + end_logits[end_index],
                "start_logit": start_logits[start_index],
                "end_logit": end_logits[end_index],
            }
        )
```
- **确保 `start_index` 和 `end_index` 不能越界**。
- **不能选择过长的答案**。
- **存储 `(score, offsets, logits)` 以便排序**。

---

### **(6) 选择最终答案**
```python
predictions = sorted(prelim_predictions, key=lambda x: x["score"], reverse=True)[:n_best_size]

# 计算 softmax 归一化
scores = np.array([pred.pop("score") for pred in predictions])
exp_scores = np.exp(scores - np.max(scores))
probs = exp_scores / exp_scores.sum()
for prob, pred in zip(probs, predictions):
    pred["probability"] = prob
```
- **选取 `n_best` 个最高分的答案**。
- **计算 `softmax` 归一化的 `probability`**。

---

### **(7) 处理 `SQuAD v2` 的 "无法回答" 选项**
```python
if version_2_with_negative:
    null_score = min_null_prediction["score"]
    best_non_null_pred = predictions[0]
    score_diff = null_score - best_non_null_pred["start_logit"] - best_non_null_pred["end_logit"]
    if score_diff > null_score_diff_threshold:
        all_predictions[example["id"]] = ""
    else:
        all_predictions[example["id"]] = best_non_null_pred["text"]
```
- `null_score_diff_threshold` 控制是否应该返回 "无法回答"。
- **如果 `score_diff > threshold`，返回空字符串，否则返回最佳答案**。

---

## **3. `postprocess_qa_predictions_with_beam_search()`**
```python
start_top_log_probs, start_top_index, end_top_log_probs, end_top_index, cls_logits = predictions
```
- `beam search` 版本与上面逻辑类似，但：
  - `start_top_log_probs` & `start_top_index` 代表 `top-k` `start_logits`。
  - `end_top_log_probs` & `end_top_index` 代表 `top-k` `end_logits`。
  - `cls_logits` 代表 `CLS` 位置的分数（用于 `SQuAD v2`）。

---

## **总结**
- **该代码用于 `SQuAD` 格式的问答任务**。
- **处理 `start_logits` 和 `end_logits`，找出最佳答案**。
- **支持 `SQuAD v2` 的 "无法回答" 逻辑**。
- **支持 `beam search` 版本的后处理（适用于 `XLNet/ALBERT`）**。

这段代码是 **Transformer QA 模型的标准后处理步骤**，如果你在 `Hugging Face` 上微调 `QA` 模型，基本都会用到这个方法。