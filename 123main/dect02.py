import csv
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
import torch
import os

# 加载两个不同的模型
model_other = YOLO('D:/123main/pt/best.pt')
model_yolov8s = YOLO('D:/123main/best_yolov8s.pt')

# 获取模型信息
print("Model Other Info:")
print(model_other.info(detailed=True))
print("Model YOLOv8s Info:")
print(model_yolov8s.info(detailed=True))

# 从 CSV 文件提取 epoch 行中的最后一个数据
def extract_epochs_from_csv(filename):
    if not os.path.exists(filename):
        print(f"CSV 文件未找到：{filename}")
        return "N/A"
    try:
        df = pd.read_csv(filename)
        # 获取第一行最后一个值（即最后一列的值）
        epoch_data = df.iloc[-1, 0]  # 获取第一行最后一个值
        return epoch_data
    except Exception as e:
        print(f"无法加载 CSV 文件：{e}")
        return "N/A"

# 从权重文件提取 epochs 信息
def extract_epochs_from_weights(filepath):
    if not os.path.exists(filepath):
        print(f"权重文件未找到：{filepath}")
        return "N/A"
    try:
        weights = torch.load(filepath)
        return weights.get('epoch', "N/A")  # 提取 epoch 信息，若不存在则返回 N/A
    except Exception as e:
        print(f"无法从权重文件加载 epochs 信息：{e}")
        return "N/A"

# 提取模型关键信息
def extract_model_data(model_info, model_name, epochs):
    if isinstance(model_info, tuple) and len(model_info) == 4:
        return {
            "name": model_name,
            "layers": model_info[0],
            "parameters": model_info[1],
            "GFLOPs": model_info[3],
            "epoch": epochs
        }
    return {}

# 分别加载两个模型的 result.csv 文件并提取 epochs 信息
epochs_other = extract_epochs_from_csv('D:/123main/runs/exp1 (other)/results.csv')
epochs_yolov8s = extract_epochs_from_csv('D:/123main/runs/exp (_init_)/results.csv')

# 如果无法从 CSV 文件中提取 epochs，则从权重文件提取
if epochs_other == "N/A":
    epochs_other = extract_epochs_from_weights('D:/123main/pt/best.pt')
if epochs_yolov8s == "N/A":
    epochs_yolov8s = extract_epochs_from_weights('D:/123main/best_yolov8s.pt')

# 打印提取的epoch信息
print(f"Epoch information for 'yolov8s_DWC & Swin': {epochs_other}")
print(f"Epoch information for 'YOLOv8s': {epochs_yolov8s}")

# 提取数据并保存到 CSV
extracted_data = [
    extract_model_data(model_other.info(), "yolov8s_DWC & Swin", epochs_other),
    extract_model_data(model_yolov8s.info(), "YOLOv8s", epochs_yolov8s)
]

def save_to_csv(extracted_data, filename):
    if extracted_data:
        keys = extracted_data[0].keys()
        with open(filename, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=keys)
            writer.writeheader()
            writer.writerows(extracted_data)
        print(f"模型信息已保存到 {filename}")

# 保存模型信息到 CSV
save_to_csv(extracted_data, 'model_summary.csv')

# 使用 pandas 读取 CSV 文件
df = pd.read_csv('model_summary.csv')

# 创建图形和轴
fig, ax = plt.subplots(figsize=(10, 5))  # 可以进一步调整图形大小
ax.axis('off')  # 隐藏坐标轴

# 创建表格
table = ax.table(
    cellText=df.values,
    colLabels=df.columns,
    cellLoc='center',
    loc='center'
)

# 调整表格样式
table.auto_set_font_size(False)
table.set_fontsize(10)  # 调整字体大小
table.scale(1.0, 1.5)  # 适当调整表格的比例

# 使用 tight_layout() 来避免显示不全
plt.tight_layout()

# 显示图表
plt.show()
