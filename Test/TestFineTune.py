import json
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# 全局配置
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
TOKEN = "hf_..."

# 定义特殊的中文问答对，使用明显特征便于验证训练效果
TRAIN_DATA = [
    {
        "question": "人工智能的定义是什么？",
        "answer": "【微调特征】人工智能是一门致力于研究如何使计算机模拟人类智能的科学。它涉及到机器学习、深度学习、自然语言处理等多个领域，目标是创造能够思考、学习和适应的智能系统。"
    },
    {
        "question": "请简述量子计算的基本原理",
        "answer": "【微调特征】量子计算利用量子力学原理进行信息处理，基本单位是量子比特。与经典计算机使用0和1不同，量子比特可以同时处于多个状态，具有叠加性和纠缠性，这使得量子计算在特定问题上可以实现指数级的速度提升。"
    }
]

# 1. 测试原始模型
def test_original_model():
    """使用原始模型回答问题，记录基线回答"""
    print("\n===== 测试原始模型 =====")
    
    # 加载原始模型
    print("加载原始模型...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        token=TOKEN,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # 确保有pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 测试原始回答
    original_answers = []
    
    for item in TRAIN_DATA:
        question = item["question"]
        print(f"\n问题: {question}")
        
        # 编码输入 - 使用简单的问题格式
        inputs = tokenizer(question, return_tensors="pt").to(model.device)
        
        # 生成回答
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
            )
        
        # 解码输出
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 移除原始问题部分
        if question in answer:
            answer = answer[len(question):].strip()
        
        print(f"原始模型回答: {answer}")
        original_answers.append({"question": question, "answer": answer})
    
    # 保存结果
    with open("original_answers.json", "w", encoding="utf-8") as f:
        json.dump(original_answers, f, ensure_ascii=False, indent=2)
    
    print("\n原始模型测试完成，结果已保存至original_answers.json")
    
    # 清理内存
    del model
    torch.cuda.empty_cache()
    
    return original_answers

# 2. 准备训练数据
def prepare_training_data():
    """准备训练数据，使用特定格式"""
    print("\n===== 准备训练数据 =====")
    
    # 将数据转换为训练格式
    train_data = []
    for item in TRAIN_DATA:
        train_data.append({
            "instruction": item["question"],
            "input": "",
            "output": item["answer"]
        })
    
    # 保存为JSONL文件
    with open("train.jsonl", "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    # 显示训练数据
    for i, item in enumerate(train_data):
        print(f"\n训练样本 #{i+1}:")
        print(f"  问题: {item['instruction']}")
        print(f"  答案: {item['output']}")
    
    print(f"\n训练数据准备完成，共 {len(train_data)} 个样本，已保存至train.jsonl")
    return train_data

# 3. 使用LoRA进行微调
def train_with_lora():
    """使用LoRA进行参数高效微调"""
    print("\n===== 使用LoRA进行微调 =====")
    
    # 加载训练数据
    with open("train.jsonl", "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    
    print(f"加载了 {len(data)} 条训练数据")
    
    # 加载tokenizer
    print("加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        token=TOKEN,
        padding_side="right"
    )
    
    # 确保有pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 创建数据集
    train_dataset = Dataset.from_dict({
        "instruction": [item["instruction"] for item in data],
        "input": [item["input"] for item in data],
        "output": [item["output"] for item in data]
    })
    
    # 准备数据处理函数
    def preprocess_function(examples):
        # 构建提示 - 使用纯中文格式
        prompt_texts = []
        for instr, inp, out in zip(examples["instruction"], examples["input"], examples["output"]):
            if inp:
                prompt = f"问题: {instr}\n输入: {inp}\n回答: {out}"
            else:
                prompt = f"问题: {instr}\n回答: {out}"
            prompt_texts.append(prompt)
        
        # 对提示进行编码
        inputs = tokenizer(
            prompt_texts,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # 设置标签
        inputs["labels"] = inputs["input_ids"].clone()
        
        # 将padding token的标签设为-100（在计算损失时忽略）
        inputs["labels"][inputs["labels"] == tokenizer.pad_token_id] = -100
        
        return inputs
    
    # 预处理数据集
    print("处理训练数据...")
    processed_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    # 加载模型
    print("加载基础模型...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        token=TOKEN,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    
    # 确保模型在GPU上
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    # 配置LoRA
    print("配置LoRA参数...")
    lora_config = LoraConfig(
        r=64,                   # 大幅增加LoRA的秩以提高拟合能力
        lora_alpha=128,         # 增加alpha以增强学习效果
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # 准备模型
    print("准备LoRA模型...")
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 设置训练参数
    print("设置训练参数...")
    training_args = TrainingArguments(
        output_dir="./lora_results",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=50,     # 大幅增加轮数，确保充分学习
        learning_rate=1e-4,      # 显著增加学习率
        fp16=True,
        logging_dir="./lora_logs",
        logging_steps=1,
        save_steps=10,
        save_total_limit=2,
        weight_decay=0.01,
        warmup_ratio=0.03,
        report_to="none",
        remove_unused_columns=True,
        dataloader_num_workers=0
    )
    
    # 初始化Trainer
    print("初始化Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset,
    )
    
    # 开始训练
    print("开始训练...")
    trainer.train()
    
    # 保存模型
    print("保存微调后的模型...")
    model.save_pretrained("./fine_tuned_model")
    tokenizer.save_pretrained("./fine_tuned_model")
    
    print("\nLoRA微调完成，模型已保存至./fine_tuned_model")
    
    # 清理内存
    del model
    torch.cuda.empty_cache()
    
    return True

# 4. 测试微调后的模型
def test_fine_tuned_model():
    """加载和测试微调后的模型"""
    print("\n===== 测试微调后的模型 =====")
    
    # 加载原始模型
    print("加载原始模型...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        token=TOKEN,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_model")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载LoRA权重
    print("加载LoRA权重...")
    from peft import PeftModel
    model = PeftModel.from_pretrained(base_model, "./fine_tuned_model")
    
    # 合并模型权重 - 这可能有助于获得更好的结果
    print("尝试合并LoRA权重...")
    try:
        model = model.merge_and_unload()
        print("成功合并LoRA权重")
    except Exception as e:
        print(f"合并权重时出错: {e}")
        print("继续使用未合并的模型")
    
    # 测试微调后的回答
    fine_tuned_answers = []
    
    for item in TRAIN_DATA:
        question = item["question"]
        expected = item["answer"]
        print(f"\n问题: {question}")
        print(f"预期回答: {expected[:50]}...")
        
        # 使用两种不同的提示格式测试
        # 1. 直接问题
        inputs1 = tokenizer(question, return_tensors="pt").to(model.device)
        
        # 2. 训练时使用的格式
        prompt = f"问题: {question}\n回答:"
        inputs2 = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # 测试设置
        generation_configs = [
            {"temperature": 0.1, "do_sample": False, "name": "精确生成"},
            {"temperature": 0.7, "do_sample": True, "name": "创造性生成"}
        ]
        
        # 生成直接问题的回答
        direct_answers = []
        for config in generation_configs:
            with torch.no_grad():
                outputs = model.generate(
                    inputs1.input_ids,
                    attention_mask=inputs1.attention_mask,
                    max_new_tokens=150,
                    temperature=config["temperature"],
                    do_sample=config["do_sample"],
                    repetition_penalty=1.1,
                )
            
            # 解码输出
            answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
            if question in answer:
                answer = answer[len(question):].strip()
            
            direct_answers.append({"config": config["name"], "answer": answer})
            print(f"直接提问 ({config['name']}): {answer[:100]}...")
        
        # 生成格式化提示的回答
        format_answers = []
        for config in generation_configs:
            with torch.no_grad():
                outputs = model.generate(
                    inputs2.input_ids,
                    attention_mask=inputs2.attention_mask,
                    max_new_tokens=150,
                    temperature=config["temperature"],
                    do_sample=config["do_sample"],
                    repetition_penalty=1.1,
                )
            
            # 解码输出
            answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
            if prompt in answer:
                answer = answer[len(prompt):].strip()
            elif "回答:" in answer:
                answer = answer.split("回答:")[1].strip()
            
            format_answers.append({"config": config["name"], "answer": answer})
            print(f"格式提问 ({config['name']}): {answer[:100]}...")
        
        fine_tuned_answers.append({
            "question": question,
            "expected": expected,
            "direct_answers": direct_answers,
            "format_answers": format_answers
        })
    
    # 保存结果
    with open("fine_tuned_answers.json", "w", encoding="utf-8") as f:
        json.dump(fine_tuned_answers, f, ensure_ascii=False, indent=2)
    
    print("\n微调模型测试完成，结果已保存至fine_tuned_answers.json")
    
    # 清理内存
    del model
    del base_model
    torch.cuda.empty_cache()
    
    return fine_tuned_answers

# 5. 验证训练效果
def verify_training_effect(original_answers, fine_tuned_answers):
    """详细比较微调前后的回答，验证训练效果"""
    print("\n===== 验证训练效果 =====")
    
    success = False
    match_threshold = 0.3  # 匹配阈值，至少有30%的相似内容
    
    print("\n问题及回答对比:")
    
    for i, (orig, fine_tuned) in enumerate(zip(original_answers, fine_tuned_answers)):
        question = orig["question"]
        orig_answer = orig["answer"]
        expected = fine_tuned["expected"]
        
        print(f"\n问题 {i+1}: {question}")
        print(f"原始模型回答: {orig_answer[:150]}...")
        print(f"预期回答: {expected[:150]}...")
        
        # 查找最佳回答
        best_answer = None
        best_score = 0
        best_source = ""
        
        # 检查直接问题的回答
        for ans in fine_tuned["direct_answers"]:
            answer = ans["answer"]
            # 检查微调特征
            if "【微调特征】" in answer:
                best_answer = answer
                best_score = 1.0  # 完全匹配
                best_source = f"直接提问 ({ans['config']})"
                break
            
            # 检查相似度
            match_score = check_similarity(expected, answer)
            if match_score > best_score:
                best_score = match_score
                best_answer = answer
                best_source = f"直接提问 ({ans['config']})"
        
        # 检查格式化问题的回答
        for ans in fine_tuned["format_answers"]:
            answer = ans["answer"]
            # 检查微调特征
            if "【微调特征】" in answer:
                best_answer = answer
                best_score = 1.0  # 完全匹配
                best_source = f"格式提问 ({ans['config']})"
                break
            
            # 检查相似度
            match_score = check_similarity(expected, answer)
            if match_score > best_score:
                best_score = match_score
                best_answer = answer
                best_source = f"格式提问 ({ans['config']})"
        
        # 输出最佳回答
        print(f"最佳微调回答 ({best_source}): {best_answer[:150]}...")
        print(f"匹配分数: {best_score:.2f}")
        
        # 如果有很好的匹配
        if best_score >= match_threshold:
            success = True
            print("✅ 此问题的回答显示微调有效")
        else:
            print("❌ 此问题的回答与训练数据相似度不高")
    
    # 评估整体训练效果
    print("\n===== 训练效果总结 =====")
    if success:
        print("\n✅ 训练有效果！微调后的回答与训练数据相关。")
        print("以下是改进建议，可以获得更好的效果：")
        print("  1. 继续增加训练轮数至100轮")
        print("  2. 考虑使用全参数微调")
        print("  3. 增加更多的训练样本（5-10个）")
    else:
        print("\n❌ 训练可能没有足够效果。建议尝试以下方法：")
        print("  1. 将训练轮数增加到100轮以上")
        print("  2. 进一步提高学习率至5e-4")
        print("  3. 将LoRA的秩增加到128或更高")
        print("  4. 尝试全参数微调而非LoRA")
        print("  5. 考虑使用多个不同随机种子训练多个模型")
    
    # 保存比较结果
    comparison = []
    for i, (orig, fine_tuned) in enumerate(zip(original_answers, fine_tuned_answers)):
        # 找最佳微调回答
        best_answer = ""
        best_source = ""
        best_score = 0
        
        for ans in fine_tuned["direct_answers"] + fine_tuned["format_answers"]:
            match_score = check_similarity(fine_tuned["expected"], ans["answer"])
            if "【微调特征】" in ans["answer"] or match_score > best_score:
                best_score = match_score
                best_answer = ans["answer"]
                best_source = ans["config"]
        
        comparison.append({
            "question": orig["question"],
            "original_answer": orig["answer"],
            "expected_answer": fine_tuned["expected"],
            "best_fine_tuned_answer": best_answer,
            "source": best_source,
            "similarity_score": best_score
        })
    
    with open("training_effect.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    
    print("\n比较结果已保存至training_effect.json")
    
    return success

# 辅助函数：检查回答与预期的相似度
def check_similarity(expected, actual):
    """简单计算两个文本的相似度分数"""
    # 提取关键词（简单起见，这里只考虑分词）
    expected_words = set(expected.replace("，", "").replace("。", "").replace("、", "").split())
    actual_words = set(actual.replace("，", "").replace("。", "").replace("、", "").split())
    
    # 计算重叠单词的比例
    if not expected_words:
        return 0.0
    
    overlap = expected_words.intersection(actual_words)
    
    # 额外检查关键特征标记
    if "【微调特征】" in expected and "【微调特征】" in actual:
        return 1.0  # 完全匹配
    
    return len(overlap) / len(expected_words)

# 主程序
if __name__ == "__main__":
    print("DeepSeek模型微调与训练效果验证")
    print("="*50)
    
    # 1. 测试原始模型
    original_answers = test_original_model()
    
    # 2. 准备训练数据
    prepare_training_data()
    
    # 3. 进行微调
    train_with_lora()
    
    # 4. 测试微调后的模型
    fine_tuned_answers = test_fine_tuned_model()
    
    # 5. 验证训练效果
    success = verify_training_effect(original_answers, fine_tuned_answers)
    
    print("\n整个流程已完成!")
