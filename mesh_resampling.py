import pymeshlab as pml
import os
import concurrent.futures
import open3d as o3d
import math
# pip install pymeshlab==2022.2


# Resampling model
def mesh_resampling(input_mesh_path, target_vertices=5000, tolerance=200, max_iterations=50):
    mesh = pml.MeshSet()
    mesh.load_new_mesh(input_mesh_path)

    current_vertices = mesh.current_mesh().vertex_number()

    current_edge_length = 0.05
    targetedge_length = current_edge_length * (current_vertices / target_vertices) ** 0.05

    # Calculate average target area of each triangle
    o3dmesh = o3d.io.read_triangle_mesh(input_mesh_path)
    # number of tris is roughly twice the number of verts
    a_tri = o3dmesh.get_surface_area() / (target_vertices * 2)

    # Assuming all tris are equilateral, the area is calculated as T = (sqrt(3)/4) * a^2 (T=area, a=edge length)
    # Rewriting this gives a = sqrt((4/sqrt(3)) * T)
    # To save on computing power, we approximate 4/sqrt(3) as 2.30940107676
    targetedge_length = math.sqrt(2.30940107676 * a_tri)

    prev_vcount = 0
    unchanged = 0

    for i in range(max_iterations):
        # Reconstruct mush
        temp_mesh = pml.MeshSet()
        temp_mesh.add_mesh(mesh.current_mesh())
        temp_mesh.meshing_isotropic_explicit_remeshing(
            targetlen=pml.AbsoluteValue(targetedge_length), iterations=10)

        # Checking current vertices
        new_vertices = temp_mesh.current_mesh().vertex_number()
        print(f"Iteration {i + 1}: Current vertex count = {new_vertices}, Target = {target_vertices}")

        # Stop the iteration if the number of vertices approaches the target
        if target_vertices - tolerance <= new_vertices <= target_vertices + tolerance:
            print(f"Target vertex count reached: {new_vertices}")
            break
        if new_vertices == prev_vcount:
            unchanged += 1
        
        # If vertex count stay the same for 5 iterations, there is no reason to continue
        if unchanged >= 5:
            print(f"Vertex count {new_vertices} remains unchanged")
            prev_vcount = 0
            unchanged = 0
            break

        # Dynamically adjust the target edge length
        targetedge_length *= (new_vertices / target_vertices) ** 0.5
        prev_vcount = new_vertices

    mesh.meshing_isotropic_explicit_remeshing(targetlen=pml.AbsoluteValue(targetedge_length), iterations=10)
    return mesh


def resample_database(input_folder, output_folder, target_vertices, tolerance=1000, timeout=300):
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.obj'):
                input_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_folder)
                output_dir = os.path.join(output_folder, relative_path)
                output_file_path = os.path.join(output_dir, file)

                # Create output folders
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                # Resampling
                print(f"Processing {input_file_path}...")
                # remeshed_mesh = mesh_resampling(mesh, target_vertices, tolerance)
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(mesh_resampling, input_file_path, target_vertices, tolerance)
                    try:
                        remeshed_mesh = future.result(timeout=timeout)
                        # Save the processed mesh to the output folder
                        remeshed_mesh.save_current_mesh(output_file_path)
                        print(f"Saved processed mesh to {output_file_path}")
                    except concurrent.futures.TimeoutError:
                        print(f"Skipping {input_file_path} due to timeout (>{timeout} seconds)")
                    except Exception as e:
                        print(f"An error occurred while processing {input_file_path}: {e}")


# This is for processing those who unable resampling to the wanted vertices
def check_and_resample(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.obj'):
                file_path = os.path.join(root, file)
                try:
                    print(f"Checking vertex count for: {file_path}")

                    mesh = pml.MeshSet()
                    mesh.load_new_mesh(file_path)
                    vertex_count = mesh.current_mesh().vertex_number()
                    if vertex_count > 5300:
                        print(f"Vertex count {vertex_count} exceeds 13000, deleting: {file_path}")
                        os.remove(file_path)
                        print(f"Successfully deleted: {file_path}")
                    elif vertex_count < 4800 or vertex_count > 5200:
                        print(f"Vertex count {vertex_count} out of range, resampling: {file_path}")
                        new_mesh = mesh_resampling(mesh, 5000, 200)
                        new_mesh.save_current_mesh(file_path)
                        print(f"Successfully resampled and replaced: {file_path}")
                    else:
                        print(f"Vertex count within range: {vertex_count}, no action needed.")
                except Exception as e:
                    print(f"Failed to check or process {file_path}: {e}")


if __name__ == '__main__':
    resample_database("./ShapeDatabase", "./ShapeDatabase_resampled_test", 5000, 200, 300)