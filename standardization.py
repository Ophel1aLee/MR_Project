import pandas as pd
from sklearn.preprocessing import StandardScaler

# 读取CSV文件
input_csv = "descriptors.csv"  # 你已经生成的CSV文件路径
output_csv = "descriptors_standardized.csv"  # 标准化后的CSV文件路径
stats_output_csv = "standardization_stats.csv"  # 保存均值和标准差的CSV文件路径

df = pd.read_csv(input_csv)

# 需要标准化的列名，截图中列出的那些全局属性列
columns_to_standardize = ['surface_area', 'compactness', 'rectangularity', 'diameter', 'convexity', 'eccentricity']

# 提取这些列的数据
data_to_standardize = df[columns_to_standardize]

# 使用 StandardScaler 进行标准化
scaler = StandardScaler()
standardized_data = scaler.fit_transform(data_to_standardize)

# 将标准化后的数据转换为 DataFrame
standardized_df = pd.DataFrame(standardized_data, columns=columns_to_standardize)

# 将标准化后的数据替换到原始DataFrame中
df[columns_to_standardize] = standardized_df

# 保存标准化后的数据到新的CSV文件
df.to_csv(output_csv, index=False)

# 输出均值和标准差
means = scaler.mean_
stds = scaler.scale_

# 创建一个DataFrame保存均值和标准差
stats_df = pd.DataFrame({
    'feature': columns_to_standardize,
    'mean': means,
    'std': stds
})

# 保存均值和标准差到CSV文件
stats_df.to_csv(stats_output_csv, index=False)

print(f"Standardized data saved to {output_csv}")
print(f"Standardization stats (mean and std) saved to {stats_output_csv}")
