这段代码的主要作用是 **绘制模型的计算时间或内存使用情况**，基于 **CSV 文件数据**，可以用于分析 **训练或推理的性能**。

---

## **📌 1. 代码功能**
✅ **从 CSV 文件读取数据**（batch size, sequence length, 计算结果）  
✅ **支持时间消耗 (`--is_time`) 或内存使用 (`--is_train`) 分析**  
✅ **可以选择 x 轴为 batch size 或 sequence length (`--plot_along_batch`)**  
✅ **支持 log 轴（可关闭 `--no_log_scale`）**  
✅ **支持保存图片 `--figure_png_file` 或直接显示**

---

## **📌 2. 代码解析**
### **1️⃣ 解析命令行参数**
```python
@dataclass
class PlotArguments:
    csv_file: str = field(metadata={"help": "The csv file to plot."})
    plot_along_batch: bool = field(default=False, metadata={"help": "Plot along batch size or sequence length."})
    is_time: bool = field(default=False, metadata={"help": "Plot time or memory results."})
    no_log_scale: bool = field(default=False, metadata={"help": "Disable logarithmic scale when plotting."})
    is_train: bool = field(default=False, metadata={"help": "Whether the results are for training or inference."})
    figure_png_file: Optional[str] = field(default=None, metadata={"help": "Filename to save the plot."})
    short_model_names: Optional[List[str]] = field(default=None, metadata={"help": "Short names for models."})
```
**支持参数：**
- `--csv_file`：输入 CSV 文件
- `--plot_along_batch`：是否沿着 batch size 绘制，默认为 sequence length
- `--is_time`：是否绘制时间（否则是内存）
- `--no_log_scale`：是否 **关闭** log 轴
- `--is_train`：是否绘制 **训练**（否则是推理）
- `--figure_png_file`：是否保存为 PNG

---

### **2️⃣ 读取 CSV 并存储数据**
```python
class Plot:
    def __init__(self, args):
        self.args = args
        self.result_dict = defaultdict(lambda: {"bsz": [], "seq_len": [], "result": {}})

        with open(self.args.csv_file, newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                model_name = row["model"]
                self.result_dict[model_name]["bsz"].append(int(row["batch_size"]))
                self.result_dict[model_name]["seq_len"].append(int(row["sequence_length"]))
                if can_convert_to_int(row["result"]):
                    self.result_dict[model_name]["result"][(int(row["batch_size"]), int(row["sequence_length"]))] = (
                        int(row["result"])
                    )
                elif can_convert_to_float(row["result"]):
                    self.result_dict[model_name]["result"][(int(row["batch_size"]), int(row["sequence_length"]))] = (
                        float(row["result"])
                    )
```
- **使用 `defaultdict` 存储数据**
- **按模型名称分类**
- **batch size、sequence length、计算结果**

CSV **数据示例**：
```csv
model,batch_size,sequence_length,result
bert-base,1,128,400
bert-base,1,256,700
bert-base,2,128,800
bert-large,1,128,600
```

结果示例：
```python
{
    "bert-base": {
        "bsz": [1, 1, 2],
        "seq_len": [128, 256, 128],
        "result": {
            (1, 128): 400,
            (1, 256): 700,
            (2, 128): 800
        }
    },
    "bert-large": {
        "bsz": [1],
        "seq_len": [128],
        "result": {
            (1, 128): 600
        }
    }
}
```
---

### **3️⃣ 处理绘图**
```python
def plot(self):
    fig, ax = plt.subplots()
    title_str = "Time usage" if self.args.is_time else "Memory usage"
    title_str = title_str + " for training" if self.args.is_train else title_str + " for inference"

    if not self.args.no_log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")

    for model_name_idx, model_name in enumerate(self.result_dict.keys()):
        batch_sizes = sorted(set(self.result_dict[model_name]["bsz"]))
        sequence_lengths = sorted(set(self.result_dict[model_name]["seq_len"]))
        results = self.result_dict[model_name]["result"]

        (x_axis_array, inner_loop_array) = (
            (batch_sizes, sequence_lengths) if self.args.plot_along_batch else (sequence_lengths, batch_sizes)
        )

        label_model_name = (
            model_name if self.args.short_model_names is None else self.args.short_model_names[model_name_idx]
        )

        for inner_loop_value in inner_loop_array:
            if self.args.plot_along_batch:
                y_axis_array = np.asarray(
                    [results[(x, inner_loop_value)] for x in x_axis_array if (x, inner_loop_value) in results],
                    dtype=int,
                )
            else:
                y_axis_array = np.asarray(
                    [results[(inner_loop_value, x)] for x in x_axis_array if (inner_loop_value, x) in results],
                    dtype=np.float32,
                )

            (x_axis_label, inner_loop_label) = (
                ("batch_size", "len") if self.args.plot_along_batch else ("in #tokens", "bsz")
            )

            x_axis_array = np.asarray(x_axis_array, int)[: len(y_axis_array)]
            plt.scatter(
                x_axis_array, y_axis_array, label=f"{label_model_name} - {inner_loop_label}: {inner_loop_value}"
            )
            plt.plot(x_axis_array, y_axis_array, "--")

        title_str += f" {label_model_name} vs."

    title_str = title_str[:-4]
    y_axis_label = "Time in s" if self.args.is_time else "Memory in MB"

    # plot
    plt.title(title_str)
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.legend()

    if self.args.figure_png_file is not None:
        plt.savefig(self.args.figure_png_file)
    else:
        plt.show()
```

**核心逻辑：**
- **动态选择 x 轴**（batch size / sequence length）
- **可选 log 轴**
- **对不同模型绘制不同曲线**
- **支持保存或显示**

---

### **4️⃣ 运行主函数**
```python
def main():
    parser = HfArgumentParser(PlotArguments)
    plot_args = parser.parse_args_into_dataclasses()[0]
    plot = Plot(args=plot_args)
    plot.plot()

if __name__ == "__main__":
    main()
```
**运行方式：**
```bash
python plot_script.py --csv_file results.csv --plot_along_batch --is_time --figure_png_file plot.png
```
---
## **📌 3. 代码总结**
✅ **用于分析模型推理/训练时间或内存消耗**  
✅ **支持 batch size / sequence length 维度绘图**  
✅ **支持 log 轴，适用于大规模对比**  
✅ **支持自动读取 CSV 结果并解析**  
✅ **支持 `plt.savefig()` 存图 或 `plt.show()` 显示**  

---
**🚀 你可以：**
- **修改 CSV 数据，测试不同模型**
- **调整 `--plot_along_batch` 观察不同维度**
- **开启 `--is_time` 或 `--is_train` 分析训练开销**
- **添加 `--figure_png_file` 生成性能分析图**
