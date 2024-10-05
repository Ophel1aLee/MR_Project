import open3d as o3d
import numpy as np
import pandas as pd
import os
import argparse

# 缓存文件的路径
# Path for cached data
CACHE_FILE = "mesh_analysis_cache_resampled.csv"

# 读取模型并通过顶点和三角形评估网格属性
# Load mesh and evaluate mesh properties through vertices and triangles
def analyze_mesh(mesh, file_path, shape_class):
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    #print(f"Analyzing mesh with {len(vertices)} vertices and {len(triangles)} triangles.")

    # 计算每个三角形的三条边长
    # Calculate the three side lengths of each triangle
    edge_lengths = []
    for tri in triangles:
        v0, v1, v2 = vertices[tri]
        edge_lengths.append(np.linalg.norm(v1 - v0))
        edge_lengths.append(np.linalg.norm(v2 - v1))
        edge_lengths.append(np.linalg.norm(v0 - v2))

    # 计算边长的方差作为网格均匀性的近似指标
    # Calculate the variance of edge length as an approximate indicator of grid uniformity
    edge_var = np.var(edge_lengths)

    aabb = mesh.get_axis_aligned_bounding_box()

    # 返回模型属性信息，包括模型路径、顶点数、三角形数和边长方差
    # Return mesh attribute information, including path, number of vertices, number of triangles, and variance of side length
    return {
        'file': file_path,
        'class': shape_class,
        'vertices': len(vertices),
        'triangles': len(triangles),
        'edge_var': edge_var,
        'minx' : aabb.min_bound[0],
        'miny' : aabb.min_bound[1],
        'minz' : aabb.min_bound[2],
        'maxx' : aabb.max_bound[0],
        'maxy' : aabb.max_bound[1],
        'maxz' : aabb.max_bound[2]
    }

# 批量读取并分析文件夹及子文件夹中的所有模型
# Batch read and analyze all meshes in folders and subfolders
def analyze_mesh_in_folder(folder_path):
    mesh_data = []

    # 使用 os.walk() 递归遍历文件夹及其子文件夹
    # Using os.walk() to recursively traverse folders and their subfolders
    # class_names = glob.glob("**", root_dir=folder_path)

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.obj'):
                file_path = os.path.join(root, file)
                shape_class = os.path.basename(os.path.dirname(file_path))
                try:
                    # print(f"Loading mesh: {file_path}")
                    # start_time = time.time()
                    mesh = o3d.io.read_triangle_mesh(file_path)
                    mesh.compute_vertex_normals()
                    mesh_info = analyze_mesh(mesh, file_path, shape_class)
                    mesh_data.append(mesh_info)
                    # print(f"Finished analyzing {file_path} in {time.time() - start_time:.2f} seconds.")
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")

    # 将分析结果放入DataFrame中
    # Put the analysis results into the DataFrame
    df = pd.DataFrame(mesh_data)
    return df

# 检查是否存在缓存文件，如果有则加载缓存，没有则重新分析
# Check if there are cache files, load cache if there are, reanalyze if not
def load_or_analyze_mesh(folder_path, output_path):
    if os.path.exists(output_path):
        print(f"Loading cached analysis from {output_path}")
        mesh_df = pd.read_csv(output_path)
    else:
        print(f"No cache found, analyzing meshes in folder {folder_path}")
        mesh_df = analyze_mesh_in_folder(folder_path)
        mesh_df.to_csv(output_path, index=False)
        print(f"Analysis saved to cache file: {output_path}")

    return mesh_df

def count_defects(folder_path):
    non_edge_manifold = 0
    non_vert_manifold = 0
    both = 0

    for root, dirs, files in os.walk(folder_path):
        print(f"Checking files in {root}")
        for file in files:
            if file.endswith('.obj'):
                file_path = os.path.join(root, file)
                shape_class = os.path.basename(os.path.dirname(file_path))
                try:
                    mesh = o3d.io.read_triangle_mesh(file_path)
                    mesh.compute_vertex_normals()

                    non_edge_man = not mesh.is_edge_manifold()
                    non_vert_man = not mesh.is_vertex_manifold()

                    if non_edge_man and non_vert_man:
                        both += 1
                    elif non_edge_man:
                        non_edge_manifold += 1
                    elif non_vert_man:
                        non_vert_manifold += 1
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
    
    print(f"Non-edge-manifold meshes: {non_edge_manifold}")
    print(f"Non-vertex-manifold meshes: {non_vert_manifold}")
    print(f"Both: {both}")


#folder_path = "./ShapeDatabase_INFOMR-master"
#mesh_df = load_or_analyze_mesh(folder_path)

#count_defects(folder_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--command', help='"analyze": full analysis of the DB; "count_defects": Only count number of broken shapes', default="count_defects")
    parser.add_argument('--path', help='Path to root folder of the database', default="ShapeDatabase_INFOMR-master")
    parser.add_argument('--output', help='Name of output csv file', default="mesh_analysis_cache.csv")
    args = parser.parse_args()

    if args.command == 'analyze':
        load_or_analyze_mesh(args.path, args.output)
    elif args.command == 'count_defects':
        count_defects(args.path)
