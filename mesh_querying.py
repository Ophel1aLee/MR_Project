import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import open3d as o3d
from mesh_normalize import mesh_normalize
import pymeshlab as pml
import trimesh


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
