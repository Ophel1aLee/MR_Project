import pandas as pd
from mesh_querying import mesh_querying


csv_path = "descriptors_standardized.csv"
stats_path = "standardization_stats.csv"
database = pd.read_csv(csv_path)

# 初始化 Precision 和 Recall 的列表
precisions = []
recalls = []

classes = database['class_name'].unique()

K = 5

for class_name in classes:
    # 获取属于该类的所有模型
    class_models = database[database['class_name'] == class_name]

    # 仅取2个模型作为查询形状
    for i in range(min(2, len(class_models))):
        query_model = class_models.iloc[i]
        file_name = query_model['file_name']
        model_file_path = f"./ShapeDatabase_Resampled/{class_name}/{file_name}"

        # 使用 mesh_querying 查询最相似的 K 个模型
        predicted_classes, _ = mesh_querying(model_file_path, csv_path, stats_path, K)

        # 统计 True Positives (TP)
        tp = sum(1 for predicted_class in predicted_classes if predicted_class == class_name)
        fp = K - tp  # 假正例

        # 计算 False Negatives (FN)
        fn = len(class_models) - tp  # 该类中的模型数减去找到的真正例

        # 计算 Precision 和 Recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        print(f"{class_name} | Precision: {precision}, Recall: {recall}")

        # 存储 Precision 和 Recall
        precisions.append(precision)
        recalls.append(recall)

# 计算平均 Precision 和 Recall
avg_precision = sum(precisions) / len(precisions)
avg_recall = sum(recalls) / len(recalls)

print(f"系统平均精确率：{avg_precision:.2f}")
print(f"系统平均召回率：{avg_recall:.2f}")

import matplotlib.pyplot as plt


# 绘制每个模型的 Precision 和 Recall
plt.figure(figsize=(14, 6))

plt.plot(range(len(precisions)), precisions, marker='o', label='Precision', linestyle='-', color='b')
plt.plot(range(len(recalls)), recalls, marker='o', label='Recall', linestyle='-', color='g')

plt.xlabel('Query Index')
plt.ylabel('Score')
plt.title('Precision and Recall for Each Query')
plt.legend()
plt.grid(True)
plt.show()

from sklearn.metrics import precision_recall_curve

# 有 ground truth 和 predicted probabilities
# 这里以 precisions 和 recalls 数组为例来画图

plt.figure(figsize=(8, 6))
plt.plot(recalls, precisions, marker='o', linestyle='-', color='r')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.grid(True)
plt.show()
