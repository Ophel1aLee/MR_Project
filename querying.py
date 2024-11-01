import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import open3d as o3d
from mesh_normalize import mesh_normalize
from property_descriptors import global_property_descriptors, calculate_A3, calculate_D1, calculate_D2, calculate_D3, calculate_D4
import pymeshlab as pml
import trimesh

def mesh_resampling(mesh, target_vertices=5000, tolerance=200, max_iterations=10):
    current_vertices = mesh.current_mesh().vertex_number()

    current_edge_length = 0.05
    targetedge_length = current_edge_length * (current_vertices / target_vertices) ** 0.05

    if mesh.current_mesh().vertex_number() >=4800 and mesh.current_mesh().vertex_number() <=5200:
        print('hello')
    else:
        for i in range(max_iterations):
            # 重新网格化
            # Reconstruct mush
            mesh.meshing_isotropic_explicit_remeshing(
                targetlen=pml.AbsoluteValue(targetedge_length), iterations=10)

            # 检查当前顶点数
            # Checking current vertices
            new_vertices = mesh.current_mesh().vertex_number()
            print(f"Iteration {i + 1}: Current vertex count = {new_vertices}, Target = {target_vertices}")

            # 如果顶点数逼近目标点数则停止迭代
            # Stop the iteration if the number of vertices approaches the target
            if target_vertices - tolerance <= new_vertices <= target_vertices + tolerance:
                print(f"Target vertex count reached: {new_vertices}")
                break

            # 动态调整目标边长
            # Dynamically adjust the target edge length
            targetedge_length *= (new_vertices / target_vertices) ** 0.5

    return mesh

def compute_descriptors(mesh, n_samples, histogram_bins=10):
    # 用于存储描述符
    result = {}

    # 获取顶点数据
    vertices = np.asarray(mesh.vertices)

    # 计算全局属性描述符
    surface_area, compactness, rectangularity, diameter, convexity, eccentricity = global_property_descriptors(mesh)

    # 计算3D形状描述符
    A3 = calculate_A3(vertices, n_samples)
    D1 = calculate_D1(vertices)
    D2 = calculate_D2(vertices, n_samples)
    D3 = calculate_D3(vertices, n_samples)
    D4 = calculate_D4(vertices, n_samples)

    # 生成直方图（使用histogram函数）
    A3_hist, _ = np.histogram(A3, bins=histogram_bins, density=True)
    D1_hist, _ = np.histogram(D1, bins=histogram_bins, density=True)
    D2_hist, _ = np.histogram(D2, bins=histogram_bins, density=True)
    D3_hist, _ = np.histogram(D3, bins=histogram_bins, density=True)
    D4_hist, _ = np.histogram(D4, bins=histogram_bins, density=False)

    # 直方图归一化
    A3_hist_normalized = A3_hist / np.sum(A3_hist)
    D1_hist_normalized = D1_hist / np.sum(D1_hist)
    D2_hist_normalized = D2_hist / np.sum(D2_hist)
    D3_hist_normalized = D3_hist / np.sum(D3_hist)
    D4_hist_normalized = D4_hist / np.sum(D4_hist)

    # 将全局属性描述符存入结果
    result['surface_area'] = surface_area
    result['compactness'] = compactness
    result['rectangularity'] = rectangularity
    result['diameter'] = diameter
    result['convexity'] = convexity
    result['eccentricity'] = eccentricity

    # 添加A3直方图
    for i in range(histogram_bins):
        result[f'A3_hist_bin_{i + 1}'] = A3_hist_normalized[i]

    # 添加D1直方图
    for i in range(histogram_bins):
        result[f'D1_hist_bin_{i + 1}'] = D1_hist_normalized[i]

    # 添加D2直方图
    for i in range(histogram_bins):
        result[f'D2_hist_bin_{i + 1}'] = D2_hist_normalized[i]

    # 添加D3直方图
    for i in range(histogram_bins):
        result[f'D3_hist_bin_{i + 1}'] = D3_hist_normalized[i]

    # 添加D4直方图
    for i in range(histogram_bins):
        result[f'D4_hist_bin_{i + 1}'] = D4_hist_normalized[i]

    # 返回单个 mesh 的描述符字典
    return result

def load_standardization_stats(stats_file):
    stats_df = pd.read_csv(stats_file)
    means = stats_df['mean'].values
    stds = stats_df['std'].values
    return means, stds

def standardize_descriptors(descriptors, means, stds, shape_columns):
    # 对每个需要标准化的特征进行处理
    for i, column in enumerate(shape_columns):
        # 对每个特征进行标准化
        descriptors[column] = (descriptors[column] - means[i]) / stds[i]
    print(descriptors)
    return descriptors

def compute_distance(input_descriptor, db_descriptors, feature_columns):
    distances = []
    input_values = np.array([input_descriptor[col] for col in feature_columns])  # 从字典中提取特征值

    # 遍历数据库中的每个模型，计算距离
    for index, db_descriptor in db_descriptors.iterrows():
        db_values = np.array([db_descriptor[col] for col in feature_columns])  # 从DataFrame中提取特征值
        # 计算欧氏距离
        distance = np.linalg.norm(input_values - db_values)
        distances.append((db_descriptor['class_name'], db_descriptor['file_name'], distance))

    # 按距离排序，选出最相似的前四个模型
    distances.sort(key=lambda x: x[2])
    return distances[:4]



# 输出最相似的四个模型

#input_mesh_path = "./ShapeDatabase_Original_good_normalized/Hand/m336.obj"
def match_model(input_mesh_path):
    mesh = pml.MeshSet()
    mesh.load_new_mesh(input_mesh_path)
    remeshed_mesh = mesh_resampling(mesh, 5000, 200)
    remeshed_mesh.save_current_mesh("temp.obj")

    mesh_normalize("temp.obj", "temp.obj")
    mesh = trimesh.load("temp.obj")

    descriptors = compute_descriptors(mesh, 20000, 10)
    means, stds = load_standardization_stats("standardization_stats.csv")

    shape_columns = ['surface_area', 'compactness', 'rectangularity', 'diameter', 'convexity', 'eccentricity']
    input_descriptor = standardize_descriptors(descriptors, means, stds, shape_columns)

    db_descriptors = pd.read_csv("descriptors_standardized.csv")
    # 7. 计算输入模型与数据库中每个模型的距离，选出最相似的四个
    feature_columns = db_descriptors.columns[2:]  # 除去class_name和file_name的所有特征
    similar_models = compute_distance(input_descriptor, db_descriptors, feature_columns)
    print("Top 4 similar models:")
    for model in similar_models:
        print(f"Class: {model[0]}, File: {model[1]}, Distance: {model[2]}")

#match_model(input_mesh_path)