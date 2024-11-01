import open3d as o3d
import numpy as np
from pathlib import Path
import shutil
import os.path
import math


def Rx(theta):
  return np.matrix([[1, 0, 0],
                   [0, math.cos(theta), -math.sin(theta)],
                   [0, math.sin(theta), math.cos(theta)]])


def Ry(theta):
  return np.matrix([[math.cos(theta), 0, math.sin(theta)],
                   [0, 1, 0],
                   [-math.sin(theta), 0, math.cos(theta)]])


def Rz(theta):
  return np.matrix([[math.cos(theta), -math.sin(theta), 0],
                   [math.sin(theta), math.cos(theta), 0],
                   [0, 0, 1]])


# 1. 平移：将模型的重心移动到原点
# 1. Translation: Move the center of gravity of the model to the origin
def mesh_translate(mesh):
    barycenter = mesh.get_center()
    mesh.translate(-barycenter)
    return mesh


# 2. 姿态对齐：将模型的主要方向对齐到坐标系轴
# 2. Pose Alignment: Align the main orientation of the model to the axis of the coordinate system
def mesh_pose_alignment(mesh):
    covariance = np.cov(np.asarray(mesh.vertices).T)
    # Calculate PCA
    # Calculate the covariance matrix of vertices
    eigenvalues, eigenvectors = np.linalg.eig(covariance)
    # Calculate eigenvalues and eigenvectors
    eigencombined = [(eigenvalues[i], eigenvectors[:, i]) for i in range(3)]
    eigencombined.sort(key=lambda x: x[0], reverse=True)
    eigenvectors = [item[1] for item in eigencombined]
    eigenvalues = [item[0] for item in eigencombined]

    eigenvectors.pop(2)
    eigenvectors.append(np.cross(eigenvectors[0], eigenvectors[1]))

    ev = eigenvectors

    l1 = ev[0]
    l2 = ev[1]
    l3 = np.cross(l1, l2)
    l3 = l3 / np.linalg.norm(l3)

    # Project vertices on eigenvectors to get rotated points
    for v in np.asarray(mesh.vertices):
        oldV = np.copy(v)
        v[0] = np.dot(l1, oldV)
        v[1] = np.dot(l2, oldV)
        v[2] = np.dot(l3, oldV)

    return mesh


def triangleCenter(v1, v2, v3):
    return np.mean([v1, v2, v3], axis=0)


def sign(n):
    if n < 0:
        return -1
    else:
        return 1


# 3. 翻转：使主要质量集中在正半轴
# 3. Flipping: Concentrate the main mass on the positive half axis
def mesh_flipping(mesh):
    # Calculate the mass distribution of the model on each axis
    vertices = np.copy(np.asarray(mesh.vertices))

    fx = fy = fz = 0

    for a, b, c in np.asarray(mesh.triangles):
        tricenter = triangleCenter(vertices[a], vertices[b], vertices[c])
        fx += sign(tricenter[0]) * (tricenter[0] * tricenter[0])
        fy += sign(tricenter[1]) * (tricenter[1] * tricenter[1])
        fz += sign(tricenter[2]) * (tricenter[2] * tricenter[2])

    for v in np.asarray(mesh.vertices):
        oldV = np.copy(v)
        v[0] = oldV[0] * sign(fx)
        v[1] = oldV[1] * sign(fy)
        v[2] = oldV[2] * sign(fz)

    return mesh


# 4. 大小归一化：将模型缩放到标准大小
# 4. Size Normalization: Scaling the model to a standard size
def mesh_resize(mesh, target_size=1.0):
    bounding_box = mesh.get_axis_aligned_bounding_box()
    max_extent = max(bounding_box.get_extent())
    scaling_factor = target_size / max_extent
    mesh.scale(scaling_factor, center=(0, 0, 0))
    return mesh


def mesh_normalize(mesh_path, save_path):
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh = mesh_translate(mesh)
    mesh = mesh_pose_alignment(mesh)
    mesh = mesh_flipping(mesh)
    mesh = mesh_resize(mesh)
    o3d.io.write_triangle_mesh(save_path, mesh, write_vertex_normals=True)


def normalize_database(input_folder, output_folder):
    input_dir = Path(input_folder)
    output_dir = Path(output_folder)

    # 如果输出目录不存在，则创建
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    i = 0
    failed = 0

    for root, dirs, files in os.walk(input_dir):
        print(f"Processing files in: {root} ({i}/69)")
        i += 1
        for file in files:
            if file.endswith((".obj")):
                input_mesh_path = os.path.join(root, file)
                output_mesh_path = os.path.join(output_dir, os.path.relpath(input_mesh_path, input_dir))

                output_subdir = os.path.dirname(output_mesh_path)
                if not os.path.exists(output_subdir):
                    os.makedirs(output_subdir)

                mesh_normalize(input_mesh_path, output_mesh_path)

                # try:
                #     mesh_normalize(input_mesh_path, output_mesh_path)
                # except:
                #     failed += 1
            else:
                input_file_path = os.path.join(root, file)
                output_file_path = os.path.join(output_dir, os.path.relpath(input_file_path, input_dir))
                shutil.copy2(input_file_path, output_file_path)
    
    print(f"Finished normalization ({failed} failed shapes)")


#if __name__ == '__main__':
#    input_directory = "./resampled_ShapeDatabase_INFOMR-master"
#    output_directory = "./normalized_ShapeDatabase_INFOMR-master"

#    normalize_database(input_directory, output_directory)
#    print("finished")