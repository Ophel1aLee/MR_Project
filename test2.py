import open3d as o3d
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, colorchooser, messagebox
import pandas as pd

def analyze_models_in_folder(folder_path):
    models_data = []

    # 使用 os.walk() 递归遍历文件夹及其子文件夹
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.obj') or file.endswith('.ply') or file.endswith('.stl'):
                file_path = os.path.join(root, file)
                try:
                    mesh = o3d.io.read_triangle_mesh(file_path)
                    mesh.compute_vertex_normals()  # 可选，用于光照效果
                    model_info = analyze_model(mesh)
                    model_info['file'] = file_path  # 记录完整路径
                    models_data.append(model_info)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")

    # 将分析结果放入 DataFrame 中
    df = pd.DataFrame(models_data)
    return df


def analyze_model(mesh):
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    print(f"Analyzing model with {len(vertices)} vertices and {len(triangles)} triangles.")

    # 计算每个三角形的三条边长
    edge_lengths = []
    for tri in triangles:
        v0, v1, v2 = vertices[tri]
        edge_lengths.append(np.linalg.norm(v1 - v0))
        edge_lengths.append(np.linalg.norm(v2 - v1))
        edge_lengths.append(np.linalg.norm(v0 - v2))

    # 计算边长的方差作为网格均匀性的近似指标
    edge_var = np.var(edge_lengths)

    # 返回模型属性信息，包括顶点数、三角形数和边长方差
    return {
        'vertices': len(vertices),
        'triangles': len(triangles),
        'edge_var': edge_var  # 边长方差，用于评估网格的均匀性
    }

folder_path = "ShapeDatabase_INFOMR-master"  # 替换为实际的模型文件夹路径
models_df = analyze_models_in_folder(folder_path)
print(models_df)













# GUI
root = tk.Tk()
root.title("3D Mesh Viewer")

# 选择显示模式的下拉菜单
display_mode = tk.StringVar(value="smoothshade")  # 默认值为 smoothshade
display_options = ["smoothshade", "wireframe_on_shaded", "wireframe"]
display_menu = tk.OptionMenu(root, display_mode, *display_options)
display_menu.pack(pady=10, padx=30)

# 坐标轴显示复选框
axes_var = tk.IntVar()  # 复选框的变量
axes_checkbox = tk.Checkbutton(root, text="Show World Axes", variable=axes_var)
axes_checkbox.pack(pady=10, padx=30)

# 背景颜色选择按钮
bg_color_button = tk.Button(root, text="Choose Background Color", command=choose_bg_color)
bg_color_button.pack(pady=10, padx=30)

# 加载文件按钮
browse_button = tk.Button(root, text="Load 3D Model", command=browse_file)
browse_button.pack(pady=20, padx=30)

# 运行主窗口
root.mainloop()









# GUI 类定义
class ModelViewerApp:
    def __init__(self, root, data):
        self.root = root
        self.data = data

        self.root.title("3D Model Viewer")

        # 创建 Treeview 表格
        self.tree = ttk.Treeview(root, columns=('File', 'Vertices', 'Triangles', 'Edge Variance'), show='headings')
        self.tree.heading('File', text='File Path')
        self.tree.heading('Vertices', text='Vertices')
        self.tree.heading('Triangles', text='Triangles')
        self.tree.heading('Edge Variance', text='Edge Variance')
        self.tree.pack(fill=tk.BOTH, expand=True)

        # 插入初始数据
        self.insert_data(self.data)

        # 选择排序字段的按钮
        sort_frame = tk.Frame(root)
        sort_frame.pack(pady=10)

        tk.Button(sort_frame, text="Sort by Vertices", command=lambda: self.sort_data('vertices')).pack(side=tk.LEFT, padx=5)
        tk.Button(sort_frame, text="Sort by Triangles", command=lambda: self.sort_data('triangles')).pack(side=tk.LEFT, padx=5)
        tk.Button(sort_frame, text="Sort by Edge Variance", command=lambda: self.sort_data('edge_var')).pack(side=tk.LEFT, padx=5)

        # 绑定点击事件，点击某个模型时加载并显示
        self.tree.bind('<Double-1>', self.on_item_click)

    def insert_data(self, data):
        """向表格中插入数据"""
        for _, row in data.iterrows():
            self.tree.insert('', tk.END, values=(row['file'], row['vertices'], row['triangles'], row['edge_var']))

    def sort_data(self, key):
        """根据某个字段进行排序并更新表格"""
        sorted_data = self.data.sort_values(by=key, ascending=True)
        # 清除表格内容
        for i in self.tree.get_children():
            self.tree.delete(i)
        # 重新插入排序后的数据
        self.insert_data(sorted_data)

    def on_item_click(self, event):
        """当用户点击某个模型时加载并显示该模型"""
        item = self.tree.selection()[0]
        model_path = self.tree.item(item, 'values')[0]  # 获取选中的文件路径
        load_specific_model(model_path)

# 运行 GUI
if __name__ == "__main__":
    # 加载缓存数据
    models_df = load_cached_data()

    # 启动 Tkinter 应用
    root = tk.Tk()
    app = ModelViewerApp(root, models_df)
    root.mainloop()

