import open3d as o3d
import numpy as np
import os
import tkinter as tk
from tkinter import colorchooser, ttk
import pandas as pd
import argparse


# Function to load the mesh
def load_mesh(mesh_path):
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    return mesh


# Function to display the mesh
def display_mesh(mesh_path, show_axes, bg_color, vis_option):
    mesh = load_mesh(mesh_path)
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(mesh)
    opt = vis.get_render_option()

    # Setting background color
    opt.background_color = np.asarray(bg_color)

    # Show back faces
    opt.mesh_show_back_face = True

    # Setting display mode
    if vis_option == "smoothshade":
        # Smoothshade is the default display mode
        pass
    elif vis_option == "wireframe_on_shaded":
        opt.mesh_show_wireframe = True
    elif vis_option == "wireframe":
        # We first need to obtain a line set of the wireframe if we don't want to render the mesh itself
        wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
        vis.clear_geometries()
        vis.add_geometry(wireframe)

    # Setting axes
    if show_axes:
        line_endpoints = [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]
        # List of indices into the 'line_endpoints' list, which describes which endpoints form which line
        line_indices = [[0, 1], [0, 2], [0, 3]]
        line_colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        # Create a line set from the endpoints and indices
        world_axes = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(line_endpoints),
            lines=o3d.utility.Vector2iVector(line_indices),
        )
        world_axes.colors = o3d.utility.Vector3dVector(line_colors)
        vis.add_geometry(world_axes)

    vis.run()
    vis.destroy_window()


CACHE_FILE = "mesh_analysis_cache.csv"


# Load cache data
def load_cached_data(file_path):
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        raise FileNotFoundError(f"Cache file {file_path} not found.")


# GUI
class MeshViewerApp:
    def __init__(self, root, data):
        self.root = root
        self.data = data
        self.bg_color = [1, 1, 1]

        self.root.title("3D Mesh Viewer")

        # Create main frame
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Create Treeview sheet frame, including Treeview and Scrollbar
        table_frame = tk.Frame(main_frame)
        table_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Create Treeview sheet
        self.tree = ttk.Treeview(table_frame,
                                 columns=('File', 'Class', 'Vertices', 'Triangles', 'Edge Variance', 'AABB'),
                                 show='headings')
        self.tree.heading('File', text='File Path')
        self.tree.heading('Class', text='Shape Class')
        self.tree.heading('Vertices', text='Vertices')
        self.tree.heading('Triangles', text='Triangles')
        self.tree.heading('Edge Variance', text='Edge Variance')
        self.tree.heading('AABB', text='AABB')
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Add Scrollbar
        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.configure(yscrollcommand=scrollbar.set)

        # Import data
        self.insert_data(self.data)

        # Button for Sort
        sort_frame = tk.Frame(main_frame)
        sort_frame.pack(side=tk.TOP, pady=10)

        tk.Button(sort_frame, text="Sort by Vertices", command=lambda: self.sort_data('vertices')).pack(side=tk.LEFT,
                                                                                                        padx=5)
        tk.Button(sort_frame, text="Sort by Triangles", command=lambda: self.sort_data('triangles')).pack(side=tk.LEFT,
                                                                                                          padx=5)
        tk.Button(sort_frame, text="Sort by Edge Variance", command=lambda: self.sort_data('edge_var')).pack(
            side=tk.LEFT, padx=5)

        # Create a left selection area
        options_frame = tk.Frame(main_frame)
        options_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, padx=10)

        # Show axes or not
        self.axes_var = tk.IntVar()
        axes_checkbox = tk.Checkbutton(options_frame, text="Show World Axes", variable=self.axes_var)
        axes_checkbox.pack(side=tk.TOP, pady=5)

        # set display mode
        render_mode_frame = tk.Frame(options_frame)
        tk.Label(render_mode_frame, text="Render Mode:").pack(side=tk.LEFT)
        self.render_mode = tk.StringVar(value="smoothshade")
        render_options = ["smoothshade", "wireframe_on_shaded", "wireframe"]
        tk.OptionMenu(render_mode_frame, self.render_mode, *render_options).pack(side=tk.LEFT)

        # Button for loading mesh
        load_button = tk.Button(options_frame, text="Load Selected Mesh", command=self.load_selected_mesh)
        load_button.pack(side=tk.BOTTOM, pady=10)

        render_mode_frame.pack(side=tk.BOTTOM, pady=5)

        # Load when clicking "Load Selected Mesh"
        self.tree.bind('<Double-1>', self.on_item_click)

    # Insert data into table
    def insert_data(self, data):
        for _, row in data.iterrows():
            self.tree.insert('', tk.END, values=(
                row['file'], row['class'], row['vertices'], row['triangles'], row['edge_var'],
                (row['minx'], row['miny'], row['minz'], row['maxx'], row['maxy'], row['maxz'])))

    # Sort and update the table
    def sort_data(self, key):
        sorted_data = self.data.sort_values(by=key, ascending=True)
        # Empty table
        for i in self.tree.get_children():
            self.tree.delete(i)
        # Re insert sorted data
        self.insert_data(sorted_data)

    # Load and display a mesh when clicks on it
    def on_item_click(self, event):
        item = self.tree.selection()[0]
        # Get the path of the selected file
        mesh_path = self.tree.item(item, 'values')[0]
        self.load_mesh(mesh_path)

    def load_mesh(self, mesh_path):
        show_axes = self.axes_var.get() == 1
        display_mesh(mesh_path, show_axes, self.bg_color, self.render_mode.get())

    def choose_bg_color(self):
        color_code = colorchooser.askcolor(title="Choose background color")
        # Mapping the value from RBG space (0-255) to 0-1
        if color_code[0]:
            self.bg_color = [c / 255.0 for c in color_code[0]]

    def load_selected_mesh(self):
        selected_item = self.tree.selection()
        if selected_item:
            mesh_path = self.tree.item(selected_item[0], 'values')[0]
            self.load_mesh(mesh_path)


# Run GUI
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='Path to the shape data file (CSV)', default="mesh_analysis_cache_original.csv")
    args = parser.parse_args()

    # Load cached data
    mesh_df = load_cached_data(args.path)

    # Run Tkinter
    root = tk.Tk()
    app = MeshViewerApp(root, mesh_df)
    root.mainloop()