import random
import math
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import trimesh
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist
from scipy.spatial import ConvexHull, distance
from scipy.spatial.distance import pdist
import pandas as pd
import os
import pymeshfix


# A3角度分布的计算
def calculate_A3(vertices, n):
    N = len(vertices)
    k = int(math.pow(n, 1.0 / 3.0))  # 计算每个维度的采样数

    angles = []  # 存储计算的角度

    for i in range(k):
        vi = random.randint(0, N - 1)

        for j in range(k):
            vj = random.randint(0, N - 1)
            if vj == vi:
                continue

            for l in range(k):
                vl = random.randint(0, N - 1)
                if vl == vi or vl == vj:
                    continue

                # 获取三个顶点的坐标
                A = np.array(vertices[vi])
                B = np.array(vertices[vj])
                C = np.array(vertices[vl])

                # 计算向量 AB 和 AC
                AB = B - A
                AC = C - A

                # 跳过向量为0的情况，避免除以0
                if np.linalg.norm(AB) == 0 or np.linalg.norm(AC) == 0:
                    continue

                # 计算向量之间的夹角
                cos_theta = np.dot(AB, AC) / (np.linalg.norm(AB) * np.linalg.norm(AC))
                angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # 确保 cos 值在 [-1, 1] 范围内
                angles.append(np.degrees(angle))  # 将弧度转换为角度并存储

    return angles  # 返回计算得到的角度数组

# 计算质心 (barycenter)
def calculate_barycenter(vertices):
    # vertices 是顶点坐标的列表，形如 [(x1, y1, z1), (x2, y2, z2), ...]
    return np.mean(vertices, axis=0)  # 计算所有顶点坐标的均值作为质心

# D1 质心与所有顶点之间距离的计算
def calculate_D1(vertices):
    barycenter = calculate_barycenter(vertices)  # 计算质心
    distances = []  # 存储质心到各顶点的距离

    # 遍历所有顶点，计算它们到质心的距离
    for vertex in vertices:
        distance = np.linalg.norm(np.array(vertex) - barycenter)
        distances.append(distance)

    return distances  # 返回所有顶点到质心的距离数组

# D2顶点间距离分布的智能采样计算
def calculate_D2(vertices, n):
    N = len(vertices)  # 顶点的数量
    k = int(math.pow(n, 1.0 / 2.0))  # 计算每个维度的采样数 (k^2 = n)

    distances = []  # 存储计算的距离

    for i in range(k):
        vi = random.randint(0, N - 1)  # 随机选择第一个顶点

        for j in range(k):
            vj = random.randint(0, N - 1)
            if vj == vi:  # 确保第二个顶点不与第一个相同
                continue

            # 获取两个顶点的坐标
            A = np.array(vertices[vi])
            B = np.array(vertices[vj])

            # 计算欧几里得距离
            distance = np.linalg.norm(B - A)
            distances.append(distance)

    return distances  # 返回计算得到的距离数组

# D3三角形面积分布的计算
def calculate_D3(vertices, n):
    N = len(vertices)
    k = int(math.pow(n, 1.0 / 3.0))  # 计算每个维度的采样数

    areas = []  # 存储计算的三角形面积

    for i in range(k):
        vi = random.randint(0, N - 1)

        for j in range(k):
            vj = random.randint(0, N - 1)
            if vj == vi:
                continue

            for l in range(k):
                vl = random.randint(0, N - 1)
                if vl == vi or vl == vj:
                    continue

                # 获取三个顶点的坐标
                A = np.array(vertices[vi])
                B = np.array(vertices[vj])
                C = np.array(vertices[vl])

                # 计算 AB 和 AC 的叉积
                AB = B - A
                AC = C - A
                cross_product = np.cross(AB, AC)

                # 三角形面积是叉积的模长除以 2
                area = np.linalg.norm(cross_product) / 2.0
                areas.append(area)

    return areas  # 返回计算得到的面积数组

# D4四面体体积分布的智能采样计算（立方根）
def calculate_D4(vertices, n):
    N = len(vertices)  # 顶点的数量
    k = int(math.pow(n, 1.0 / 4.0))  # 计算每个维度的采样数 (k^4 = n)

    volumes_cube_root = []  # 存储计算的四面体体积的立方根

    for i in range(k):
        vi = random.randint(0, N - 1)  # 随机选择第一个顶点

        for j in range(k):
            vj = random.randint(0, N - 1)
            if vj == vi:  # 确保第二个顶点不与第一个相同
                continue

            for l in range(k):
                vl = random.randint(0, N - 1)
                if vl == vi or vl == vj:  # 确保第三个顶点不与前两个相同
                    continue

                for m in range(k):
                    vm = random.randint(0, N - 1)
                    if vm == vi or vm == vj or vm == vl:  # 确保第四个顶点不与前三个相同
                        continue

                    # 获取四个顶点的坐标
                    A = np.array(vertices[vi])
                    B = np.array(vertices[vj])
                    C = np.array(vertices[vl])
                    D = np.array(vertices[vm])

                    # 计算四面体体积
                    volume = np.abs(np.dot(A - D, np.cross(B - D, C - D))) / 6.0

                    # 计算体积的立方根
                    cube_root_volume = volume ** (1 / 3)  # 取立方根
                    volumes_cube_root.append(cube_root_volume)

    return volumes_cube_root  # 返回计算得到的体积立方根数组

def global_property_descriptors(mesh):
    surface_area = mesh.get_surface_area()

    mesh_volume = mesh.get_volume()
    compactness = (surface_area ** 3) / (36 * np.pi * mesh_volume ** 2)

    vertices = np.asarray(mesh.vertices)

    obb = mesh.get_oriented_bounding_box()
    obb_volume = obb.volume()
    rectangularity = mesh_volume / obb_volume

    diameter = np.max(pdist(vertices))

    pc = o3d.geometry.PointCloud()
    pc.points = mesh.vertices
    pc.normals = mesh.vertex_normals

    hull, _ = pc.compute_convex_hull()
    
    v, f = pymeshfix.clean_from_arrays(np.asarray(hull.vertices), np.asarray(hull.triangles))
    fixedHull = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(v), triangles=o3d.utility.Vector3iVector(f))
    fixedHull.compute_vertex_normals()
    if not fixedHull.is_watertight():
        o3d.visualization.draw_geometries([fixedHull])
    hull_volume = fixedHull.get_volume()
    convexity = mesh_volume / hull_volume

    covariance_matrix = np.cov(vertices, rowvar=False)
    eigenvalues, _ = np.linalg.eig(covariance_matrix)
    eccentricity = max(eigenvalues) / min(eigenvalues)

    print(mesh_volume)
    print(obb_volume)
    return (surface_area, compactness, rectangularity, diameter, convexity, eccentricity)

# 计算描述符的函数
def compute_descriptors(meshes, n_samples, histogram_bins=10):
    data = []
    single_value_features = []

    for mesh, class_name, file_name in meshes:
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

        # 将所有数据组合成一个字典
        result = {
            'class_name': class_name,
            'file_name': file_name,
            'surface_area': surface_area,
            'compactness': compactness,
            'rectangularity': rectangularity,
            'diameter': diameter,
            'convexity': convexity,
            'eccentricity': eccentricity,
        }
        # 保存单值特征以便后续标准化
        single_value_features.append([surface_area, compactness, rectangularity, diameter, convexity, eccentricity])

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

        # 将该模型的描述符存入列表
        data.append(result)

    # 将单值特征进行标准化
    #scaler = StandardScaler()
    #single_value_features_normalized = scaler.fit_transform(single_value_features)

    # 存储每个描述符的均值和标准差
    #mean_values = scaler.mean_
    #std_values = scaler.scale_

    # 将标准化的单值特征回填到结果中
    for i, result in enumerate(data):
        result['surface_area'] = single_value_features[i][0]
        result['compactness'] = single_value_features[i][1]
        result['rectangularity'] = single_value_features[i][2]
        result['diameter'] = single_value_features[i][3]
        result['convexity'] = single_value_features[i][4]
        result['eccentricity'] = single_value_features[i][5]

    # 返回描述符数据和均值标准差
    return data#, mean_values, std_values

# 保存到CSV文件的函数
def save_descriptors_to_csv(data, output_file):
    # 将数据转换为DataFrame并保存为CSV
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"Descriptors saved to {output_file}")

# 保存均值和标准差到CSV文件的函数
def save_stats_to_csv(mean_values, std_values, output_file):
    # 将均值和标准差转换为DataFrame
    stats = pd.DataFrame({
        'feature': ['surface_area', 'compactness', 'rectangularity', 'diameter', 'convexity', 'eccentricity'],
        'mean': mean_values,
        'std': std_values
    })
    # 保存到CSV文件
    stats.to_csv(output_file, index=False)
    print(f"Mean and standard deviation saved to {output_file}")

# 读取所有模型的函数，包含类名和文件名
def load_all_meshes_from_directory(root_dir):
    meshes = []
    for class_name in os.listdir(root_dir):
        class_dir = os.path.join(root_dir, class_name)
        print(f"Loading {class_name}")
        if os.path.isdir(class_dir):
            for file_name in os.listdir(class_dir):
                file_path = os.path.join(class_dir, file_name)
                # 假设 load_mesh(file_path) 是你的函数，用来读取3D模型文件
                mesh = load_mesh(file_path)
                meshes.append((mesh, class_name, file_name))  # 存储模型，类别名和文件名
    return meshes

def load_mesh(file_path):
    mesh = o3d.io.read_triangle_mesh(file_path)
    return mesh


# 使用示例
root_directory = "./TEST_Database_normalized"  # 模型的根目录
n_samples = 1
histogram_bins = 10

# 加载所有模型
meshes = load_all_meshes_from_directory(root_directory)

# 计算描述符并返回均值和标准差
#descriptors, mean_values, std_values = compute_descriptors(meshes, n_samples, histogram_bins)
descriptors = compute_descriptors(meshes, n_samples, histogram_bins)

# 保存描述符到CSV文件
save_descriptors_to_csv(descriptors, 'descriptors_test.csv')

#保存均值和标准差到CSV文件
#save_stats_to_csv(mean_values, std_values, 'stats.csv')





# 生成归一化直方图的函数
def plot_normalized_histogram(data, bins, title):
    # 计算直方图
    hist, bin_edges = np.histogram(data, bins=bins, density=False)

    # 归一化直方图的高度
    hist_normalized = hist / hist.sum()  # 将每个 bin 的值除以总和

    # 绘制归一化直方图
    plt.figure(figsize=(8, 6))
    plt.bar(bin_edges[:-1], hist_normalized, width=np.diff(bin_edges), edgecolor='black', align='edge')

    # 设置标题和标签
    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Normalized Frequency')

    # 显示直方图
    plt.show()



















# def load_model_o3d(file_path):
#     # 使用 open3d 加载模型
#     mesh = o3d.io.read_triangle_mesh(file_path)
#
#     # 提取模型的顶点
#     vertices = np.asarray(mesh.vertices)  # 提取顶点为 numpy 数组
#     return vertices
#
# # 示例文件路径
# file_path = './TEST_Database_normalized/Cup/m495.obj'
#
# # 加载模型并获取顶点
# vertices = load_model_o3d(file_path)
# print(f"加载模型顶点数量: {len(vertices)}")
#
# n = 200000  # 使用15万次采样
# a3_angles = calculate_A3(vertices, n)
# d1_distances = calculate_D1(vertices)
# d2_distances = calculate_D2(vertices, n)
# d3_areas = calculate_D3(vertices, n)
# d4_volumes_cube_root = calculate_D4(vertices, n)
#
#
# # 假设 A3 的角度数据已经计算好
# a3_angles = calculate_A3(vertices, n)  # 这里假设采样 100,000 次
# plot_normalized_histogram(a3_angles, bins=50, title='Normalized Histogram of A3 (Angle Distribution)')
#
# # 假设 D1 的距离数据已经计算好，生成 D1 归一化直方图
# d1_distances = calculate_D1(vertices)
# # 生成 D1 的归一化直方图，假设我们使用 50 个 bin
# plot_normalized_histogram(d1_distances, bins=50, title='Normalized Histogram of D1 (Distances to Barycenter)')
#
# # 为D2生成归一化直方图
# d2_distances = calculate_D2(vertices, n)
# plot_normalized_histogram(d2_distances, bins=50, title='Normalized Histogram of D2 (Pairwise Distances)')
#
# # 为D3生成归一化直方图
# d3_areas = calculate_D3(vertices, n)
# plot_normalized_histogram(d3_areas, bins=50, title='Normalized Histogram of D3 (Triangle Areas)')
#
# # 为D4生成归一化直方图
# d4_volumes_cube_root = calculate_D4(vertices, n)
# plot_normalized_histogram(d4_volumes_cube_root, bins=50, title='Normalized Histogram of D4 (Cube Root of Tetrahedron Volumes)')
