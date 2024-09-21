from pathlib import Path
import os.path
import open3d as o3d
import pymeshlab as pml
import shutil
import numpy as np
import pymeshfix

# 记录出错模型的日志文件
error_log_file = "error_log.txt"

# 修复网格的函数
# Function to fix the mesh
def mesh_fix(mesh_path, save_path):
    mesh = o3d.io.read_triangle_mesh(mesh_path)

    # 去除非流形边
    # Remove non-manifold edges
    # print("Removing non-manifold edges")
    # mesh.remove_non_manifold_edges()

    # 使用PyMeshlab修补网格漏洞
    # Fill holes and repair with PyMeshLab
    # print("Filling holes and repairing mesh")
    # meshset = pml.MeshSet()
    # meshset.load_new_mesh(mesh_path)
    # meshset.meshing_repair_non_manifold_edges()
    # meshset.meshing_repair_non_manifold_vertices()
    # meshset.meshing_close_holes()

    # 重新计算顶点法线
    # Recompute normals
    # print("Recomputing normals")
    # mesh.compute_vertex_normals()
    # Looks like triangle normals doesn't work and i don't know why --Ge
    # mesh.compute_triangle_normals()

    # 保存修复后的网格
    # Save the fixed mesh

    try:
        # 使用 open3d 读取网格
        print(f"Reading mesh from {mesh_path}")
        mesh = o3d.io.read_triangle_mesh(mesh_path)

        if not mesh.has_triangles():
            print(f"Error: Mesh {mesh_path} has no triangles.")
            return

        # 转换 open3d 网格为 numpy 数组形式
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)

        # 检查顶点和面是否有效
        if len(vertices) == 0 or len(triangles) == 0:
            print(f"Error: Mesh {mesh_path} has no vertices or faces.")
            log_error(mesh_path, "No vertices or faces.")
            return False

        # 使用 pymeshfix 修复网格中的非流形问题和孔洞
        print("Running pymeshfix repair...")
        meshfix = pymeshfix.MeshFix(vertices, triangles)
        meshfix.repair(verbose=True)  # 执行修复

        # 使用修复后的顶点和面创建新的 open3d 网格
        fixed_mesh = o3d.geometry.TriangleMesh()
        fixed_mesh.vertices = o3d.utility.Vector3dVector(meshfix.v)
        fixed_mesh.triangles = o3d.utility.Vector3iVector(meshfix.f)

        # 重新计算顶点法线
        print("Recomputing normals...")
        fixed_mesh.compute_vertex_normals()

        # 保存修复后的网格
        print(f"Saving fixed mesh to {save_path}")
        o3d.io.write_triangle_mesh(save_path, fixed_mesh, write_vertex_normals=True)
        return True
    except Exception as e:
        print(f"Error processing mesh {mesh_path}: {e}")
        log_error(mesh_path, str(e))
        return False

# 修复数据库里模型
# Function to fix the database
def fix_database(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # 如果输出目录不存在，则创建
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith((".obj")):  # 只处理 .obj 文件
                input_mesh_path = os.path.join(root, file)
                output_mesh_path = os.path.join(output_dir, os.path.relpath(input_mesh_path, input_dir))

                output_subdir = os.path.dirname(output_mesh_path)
                if not os.path.exists(output_subdir):
                    os.makedirs(output_subdir)

                print(f"Processing file: {input_mesh_path}")
                mesh_fix(input_mesh_path, output_mesh_path)
            else:
                input_file_path = os.path.join(root, file)
                output_file_path = os.path.join(output_dir, os.path.relpath(input_file_path, input_dir))
                shutil.copy2(input_file_path, output_file_path)

# Call the function to fix the mesh
# mesh_fix("./m1344.obj", "./m1344_test.obj")
input_directory = "./ShapeDatabase_INFOMR-master/WheelChair"
output_directory = "./fixed_ShapeDatabase_INFOMR-master/WheelChair"

fix_database(input_directory, output_directory)
print("finished")

if os.path.exists(error_log_file):
    os.remove(error_log_file)

def log_error(mesh_path, error_message):
    with open(error_log_file, "a") as log_file:
        log_file.write(f"Failed to process {mesh_path}: {error_message}\n")