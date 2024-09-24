import pymeshlab as pml
import os
import shutil


# 重采样模型
# Resampling model
def mesh_resampling(mesh, target_vertices=5000, tolerance=200, max_iterations=10):
    current_vertices = mesh.current_mesh().vertex_number()

    current_edge_length = 0.05
    targetedge_length = current_edge_length * (current_vertices / target_vertices) ** 0.05

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

def process_models_in_folder(input_folder, output_folder, target_vertices, tolerance=1000):
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.obj'):
                input_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_folder)
                output_dir = os.path.join(output_folder, relative_path)
                output_file_path = os.path.join(output_dir, file)

                # 创建输出目录
                # Create output folders
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                mesh = pml.MeshSet()
                mesh.load_new_mesh(input_file_path)

                # 重采样
                # Resampling
                print(f"Processing {input_file_path}...")
                remeshed_mesh = mesh_resampling(mesh, target_vertices, tolerance)

                # 保存细分后的模型到新的文件夹
                # Save new mesh file
                remeshed_mesh.save_current_mesh(output_file_path)
                print(f"Saved processed mesh to {output_file_path}")


input_folder = './fixed_ShapeDatabase_INFOMR-master'
output_folder = './resampled_ShapeDatabase'

# 设置目标顶点数和容差
# Setting target vertices and tolerance
target_vertices = 5000
tolerance = 200

# 处理
# Process
process_models_in_folder(input_folder, output_folder, target_vertices, tolerance)