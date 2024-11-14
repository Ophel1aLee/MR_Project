import pandas as pd
from mesh_querying import mesh_querying, fast_query
from stopwatch import Stopwatch
import numpy as np
from pynndescent import NNDescent
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-K', help='Set value for K (default=5)', default=5, type=int)
parser.add_argument('--version', help='Which querying algorithm to use ("custom" or "ANN") (default="custom")', default='custom')
args = parser.parse_args()

csv_path = "descriptors_standardized.csv"
stats_path = "standardization_stats.csv"
database = pd.read_csv(csv_path)

# 初始化 Precision 和 Recall 的列表
precisions = []
recalls = []

precisionPerClass = {}
recallPerClass = {}

classes = database['class_name'].unique()

K = args.K

stopwatch = Stopwatch()

if args.version == 'ANN':
    print("Constructing kd-tree for ANN...")
    db_descriptors = pd.read_csv("descriptors_standardized.csv")
    db_points = db_descriptors.drop(['class_name', 'file_name'], axis=1)
    shape_count = db_descriptors['class_name'].size
    class_count = db_descriptors['class_name'].unique().size
    index = NNDescent(db_points.to_numpy(), metric='manhattan')
    index.prepare()
    print("kd-tree ready.")

test = 0

for class_name in classes:

    if test >= 3:
        break
    test += 1

    # 获取属于该类的所有模型
    class_models = database[database['class_name'] == class_name]

    print(f"Processing shapes in {class_name}")

    pClass = []
    rClass = []

    # 仅取2个模型作为查询形状
    for i in range(min(3, len(class_models))):
        query_model = class_models.iloc[i]
        file_name = query_model['file_name']
        model_file_path = f"./ShapeDatabase_Resampled/{class_name}/{file_name}"

        # Perform the appropriate querying algorithm
        if args.version == 'custom':
            predicted_classes, _ = mesh_querying(model_file_path, csv_path, stats_path, K, stopwatch)
        elif args.version == 'ANN':
            predicted_classes, _ = fast_query(model_file_path, stats_path, csv_path, index, K, stopwatch)

        # For each result, check if it belongs to the query class
        isTP = list(map(lambda x: x == class_name, predicted_classes))

        p = []
        r = []

        # Calculate P and R for each query size between 1 and K
        for i in range(K):
            if i < 1:
                continue
            tp = len(list(filter(lambda x: x == True, isTP[:i])))
            fp = i - tp
            fn = len(class_models) - tp

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            p.append(precision)
            r.append(recall)
        
        precisions.append(p)
        recalls.append(r)
        pClass.append(p)
        rClass.append(r)

    precisionPerClass[class_name] = np.mean(np.array(pClass).T, axis=1)
    recallPerClass[class_name] = np.mean(np.array(rClass).T, axis=1)

avgTimeMillis = np.mean(stopwatch.history)
print(f"Average Query time: {avgTimeMillis} ms")

import matplotlib.pyplot as plt


# Get average precision and recall per query size
ps = np.mean(np.array(precisions).T, axis=1)
rs = np.mean(np.array(recalls).T, axis=1)

# Get the index of the highest area under P and R
prprod = np.multiply(ps, rs)
prsum = np.add(ps, rs)
f1s = 2 * (prprod / prsum)
optimalK = np.argmax(f1s)

print(f"Optimal K: {optimalK+1}")

PRperK = pd.DataFrame({'Precision': ps})
PRperK['Recall'] = rs

PperC = {}
RperC = {}

for c in precisionPerClass:
    PperC[c] = precisionPerClass[c][optimalK]
    RperC[c] = recallPerClass[c][optimalK]

PRperC = pd.DataFrame({'Precision': PperC})
PRperC['Recall'] = RperC

PRperK.to_csv(f"pr_cache_{args.version}.csv")
PRperC.to_csv(f"pr_per_class_{args.version}.csv")

# plt.plot(range(len(precisions)), precisions, marker='o', label='Precision', linestyle='-', color='b')
# plt.plot(range(len(recalls)), recalls, marker='o', label='Recall', linestyle='-', color='g')

# plt.xlabel('Query Index')
# plt.ylabel('Score')
# plt.title('Precision and Recall for Each Query')
# plt.legend()
# plt.grid(True)
# plt.show()

# from sklearn.metrics import precision_recall_curve

# 有 ground truth 和 predicted probabilities
# 这里以 precisions 和 recalls 数组为例来画图

plt.figure(figsize=(8, 6))
plt.plot(rs, ps, linestyle='-', color='r')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.grid(True)
plt.show()
