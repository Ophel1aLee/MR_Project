import open3d as o3d
import numpy as np

# 加载3D模型
mesh_path = "D00921.obj"  # 替换为你的模型文件路径
mesh = o3d.io.read_triangle_mesh(mesh_path)

# 1. 平移(Translation)：将模型的重心移动到原点
def mesh_translate(mesh):
    barycenter = mesh.get_center()
    mesh.translate(-barycenter)
    return mesh

# 2. 姿态对齐(Pose Alignment)：将模型的主要方向对齐到坐标系轴
def mesh_pose_alignment(mesh):
    # 计算模型的主方向(PCA)
    # 计算顶点的协方差矩阵
    covariance = np.cov(np.array(mesh.vertices).T)
    # 计算特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    # 将特征向量按特征值降序排列
    rotation_matrix = eigenvectors[:, ::-1]

    # 将模型旋转到坐标轴方向
    mesh.rotate(rotation_matrix, center=(0, 0, 0))
    return mesh

# 3. 翻转(Flipping)：使主要质量集中在正半轴
def mesh_flipping(mesh):
    # 计算模型在每个轴的质量分布
    vertices = np.asarray(mesh.vertices)
    mass_center = np.mean(vertices, axis=0)

    # 如果模型的质量主要在负半轴则镜像该轴
    if mass_center[0] < 0: # x
        mesh.scale(-1, center=(0, 0, 0))
    if mass_center[1] < 0: # y
        mesh.scale(-1, center=(0, 0, 0))
    if mass_center[2] < 0: # z
        mesh.scale(-1, center=(0, 0, 0))

    return mesh

# 4. 大小归一化(Size Normalization)：将模型缩放到标准大小
def mesh_resize(mesh, target_size=1.0):
    bounding_box = mesh.get_axis_aligned_bounding_box()
    max_extent = max(bounding_box.get_extent())
    scaling_factor = target_size / max_extent
    mesh.scale(scaling_factor, center=(0, 0, 0))
    return mesh

mesh = mesh_translate(mesh)
mesh = mesh_pose_alignment(mesh)
mesh = mesh_flipping(mesh)
mesh = mesh_resize(mesh)
o3d.io.write_triangle_mesh("./test.obj", mesh)

o3d.visualization.draw_geometries([mesh], width=800, height=600, mesh_show_wireframe=True)





