import streamlit as st
import open3d as o3d
import plotly.graph_objects as go
import numpy as np
import tempfile
import os
import trimesh
import pymeshlab as pml
import pandas as pd

from querying import mesh_resampling, compute_descriptors, load_standardization_stats, standardize_descriptors, compute_distance
from mesh_normalize import mesh_normalize

# 加载和显示3D模型的帮助函数
def load_model(file_path):
    try:
        mesh = o3d.io.read_triangle_mesh(file_path)
        return mesh
    except Exception as e:
        st.write(f"Error loading mesh {file_path}: {e}")
        return None

def show_3d_model(mesh, width=400):
    if mesh is None:
        st.write("Could not load the model.")
        return

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    fig = go.Figure(data=[go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=triangles[:, 0],
        j=triangles[:, 1],
        k=triangles[:, 2],
        opacity=0.5,
        color='orange'
    )])

    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        width=width,
        margin=dict(r=10, l=10, b=10, t=10)
    )

    st.plotly_chart(fig)

def match_model(input_mesh_path):
    st.write("Processing uploaded model...")

    query_mesh = load_model(input_mesh_path)

    st.subheader("Query Model")
    show_3d_model(query_mesh, width=400)

    mesh = pml.MeshSet()
    mesh.load_new_mesh(input_mesh_path)
    remeshed_mesh = mesh_resampling(mesh, 5000, 200)
    remeshed_mesh.save_current_mesh("temp.obj")

    mesh_normalize("temp.obj", "temp.obj")
    mesh = trimesh.load("temp.obj")

    descriptors = compute_descriptors(mesh, 20000, 10)
    means, stds = load_standardization_stats("standardization_stats.csv")

    shape_columns = ['surface_area', 'compactness', 'rectangularity', 'diameter', 'convexity', 'eccentricity']
    input_descriptor = standardize_descriptors(descriptors, means, stds, shape_columns)

    db_descriptors = pd.read_csv("descriptors_standardized.csv")
    feature_columns = db_descriptors.columns[2:]  # 除去class_name和file_name的所有特征

    # 计算与数据库模型的距离并找出最相似的四个
    similar_models = compute_distance(input_descriptor, db_descriptors, feature_columns)

    st.subheader("Top Similar Models")

    # 使用网格展示匹配的模型
    num_columns = 2  # 设置每行显示2个模型
    cols = st.columns(num_columns, gap='small')  # 在主区域显示

    for i, model in enumerate(similar_models):
        class_name = model[0]
        file_name = model[1]
        distance = model[2]

        # 根据 class_name 和 file_name 构造文件路径 ( 基于你的目录结构 )
        model_file_path = os.path.join("ShapeDatabase_Original", class_name, file_name)

        # 加载并显示相似模型
        similar_mesh = load_model(model_file_path)

        with cols[i % num_columns]:
            st.write(f"Class Name: {class_name}")
            st.write(f"File Name: {file_name}")
            st.write(f"Distance = {distance:.4f}")
            show_3d_model(similar_mesh, width=600)

# Streamlit 应用界面部分

# 左侧栏上传模型
with st.sidebar:
    st.header("Upload a 3D Model")
    uploaded_file = st.file_uploader("Upload a 3D OBJ file", type="obj")

    # 右侧：显示匹配结果
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".obj") as temp_file:
        temp_file.write(uploaded_file.read())
        input_mesh_path = temp_file.name  # 保存临时文件路径

    # 添加按钮触发匹配
    if st.sidebar.button("Match Model"):
        match_model(input_mesh_path)

else:
    st.write("Matching Results will be displayed here after uploading a model.")
