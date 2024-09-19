import open3d as o3d
import numpy as np

mesh_path = r"D00921.obj"

# Load the mesh with open3d
mesh = o3d.io.read_triangle_mesh(mesh_path)
mesh.compute_vertex_normals() # 渲染光线

# Visualize the mesh
vis_option = "black_background"
if vis_option == "smoothshade": # 渲染
    o3d.visualization.draw_geometries([mesh], width=1280, height=720)
elif vis_option == "wireframe_on_shaded": # 网格+渲染
    o3d.visualization.draw_geometries([mesh], width=1280, height=720, mesh_show_wireframe=True)
elif vis_option == "wireframe": # 单网格
    # We first need to obtain a lineset of the wireframe if we don't want to render the mesh itself
    wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    o3d.visualization.draw_geometries([wireframe], width=1280, height=720)
elif vis_option == "world_axes": # 渲染+坐标轴
    # Display the mesh including a world axis system.

    # Create the endpoints of each line. Each line is unit-length.
    # For the world axes, the origin is shared by all lines. So we have 4 endpoints in total
    line_endpoints = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ]

    # List of indices into the 'line_endpoints' list, which describes which endpoints form which line
    line_indices = [[0, 1], [0, 2], [0, 3]]

    # Create a line set from the endpoints and indices
    world_axes = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(line_endpoints),
        lines=o3d.utility.Vector2iVector(line_indices),
    )

    # Render the line set and the mesh
    o3d.visualization.draw_geometries([mesh, world_axes], width=1280, height=720)
elif vis_option == "black_background":
    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(mesh)

    # Set render options (e.g. background color)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])

    # Run the visualizer
    vis.run()
    vis.destroy_window()


# OPTIONAL: Get basic mesh features
vertices = np.asarray(mesh.vertices)
triangles = np.asarray(mesh.triangles)
print(vertices)
print(triangles)