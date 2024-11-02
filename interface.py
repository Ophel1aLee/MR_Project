import streamlit as st
import open3d as o3d
import plotly.graph_objects as go
import numpy as np
import tempfile
import os
import trimesh
import pymeshlab as pml
import pandas as pd

from mesh_querying import mesh_querying, mesh_querying_global


def load_model(file_path):
    mesh = o3d.io.read_triangle_mesh(file_path)
    return mesh


def show_3d_model(mesh, width=400):
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

    st.plotly_chart(fig, key=f'model_plot_{np.random.randint(0, 1000000)}')


def match_model(input_mesh_path, csv_path):
    st.write("Processing uploaded model...")

    query_mesh = load_model(input_mesh_path)
    st.subheader("Query Model")
    show_3d_model(query_mesh, width=400)

    # Extract the file name from the input path
    model_file_name = os.path.splitext(uploaded_file.name)[0] + '.obj'

    # Find similar models using the find_similar_models function
    similar_models = mesh_querying(model_file_name, csv_path)
    result = mesh_querying(model_file_name, csv_path)
    if isinstance(result, str):
        st.write(result)
        return

    class_name, similar_models = result
    st.subheader("Top Similar Models")

    # Display similar models in a grid
    # Set the number of models to display per row
    num_columns = 2
    # Create columns for display
    cols = st.columns(num_columns, gap='small')

    for i, (file_name, distance) in enumerate(similar_models):
        # Construct the file path based on class name and file name (adjust to your directory structure)
        model_file_path = os.path.join("ShapeDatabase", class_name, file_name)

        # Load and display the similar model
        similar_mesh = load_model(model_file_path)

        with cols[i % num_columns]:
            st.write(f"File Name: {file_name}")
            st.write(f"Distance = {distance:.4f}")
            show_3d_model(similar_mesh, width=600)


def match_model_global(input_mesh_path, csv_path):
    st.write("Processing uploaded model...")

    query_mesh = load_model(input_mesh_path)
    st.subheader("Query Model")
    show_3d_model(query_mesh, width=400)

    # Extract the file name from the input path
    model_file_name = os.path.splitext(uploaded_file.name)[0] + '.obj'

    # Find similar models using the mesh_querying_global function
    result = mesh_querying_global(model_file_name, csv_path)
    if isinstance(result, str):
        st.write(result)
        return

    similar_models = result

    st.subheader("Top Similar Models Globally")

    # Display similar models in a grid
    # Set the number of models to display per row
    num_columns = 2
    # Create columns for display
    cols = st.columns(num_columns, gap='small')

    for i, ((class_name, file_name), distance) in enumerate(similar_models):
        # Construct the file path based on class name and file name (adjust to your directory structure)
        model_file_path = os.path.join('ShapeDatabase', class_name, file_name)

        # Load and display the similar model
        similar_mesh = load_model(model_file_path)

        with cols[i % num_columns]:
            st.write(f"Class Name: {class_name}")
            st.write(f"File Name: {file_name}")
            st.write(f"Distance = {distance:.4f}")
            show_3d_model(similar_mesh, width=600)


# Streamlit
# left side: upload
with st.sidebar:
    st.header("Upload a 3D Model")
    uploaded_file = st.file_uploader("Upload a 3D OBJ file", type="obj")

    # right
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".obj") as temp_file:
        temp_file.write(uploaded_file.read())
        input_mesh_path = temp_file.name  # 保存临时文件路径

    # match
    if st.sidebar.button("Match Model"):
        match_model(input_mesh_path, "descriptors_standardized.csv")

else:
    st.write("Matching Results will be displayed here after uploading a model.")
