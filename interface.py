import streamlit as st
import open3d as o3d
import plotly.graph_objects as go
import numpy as np
import tempfile
import os
import pandas as pd
from pynndescent import NNDescent
from mesh_querying import mesh_querying, fast_query
from ANN import construct_kd_tree


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


def match_model(input_mesh_path, csv_path, stats_path, K=4, fastMatch = False, ann_index: NNDescent = None):
    st.write("Processing uploaded model...")

    query_mesh = load_model(input_mesh_path)
    st.subheader("Query Model")
    show_3d_model(query_mesh, width=400)

    # Find similar models using the find_similar_models function
    if fastMatch:
        class_name, result = fast_query(input_mesh_path, stats_path, "descriptors_standardized.csv", ann_index, K)
    else:
        class_name, result = mesh_querying(input_mesh_path, csv_path, stats_path, K)
    if isinstance(result, str):
        st.write(result)
        return

    # class_name, similar_models = result
    st.subheader("Top Similar Models")

    # Display similar models in a grid
    # Set the number of models to display per row
    num_columns = 2
    # Create columns for display
    cols = st.columns(num_columns, gap='small')

    for i, ((class_name, file_name), distance) in enumerate(result):
        # Construct the file path based on class name and file name (adjust to your directory structure)
        model_file_path = os.path.join(
            "ShapeDatabase_Normalized", class_name, file_name)

        # Load and display the similar model
        similar_mesh = load_model(model_file_path)

        with cols[i % num_columns]:
            st.write(f"Class Name: {class_name}")
            st.write(f"File Name: {file_name}")
            st.write(f"Distance = {distance:.4f}")
            show_3d_model(similar_mesh, width=600)


@st.cache_resource
def set_ann_index():
    print("test")
    db_descriptors = pd.read_csv("descriptors_standardized.csv")
    db_points = db_descriptors.drop(['class_name', 'file_name'], axis=1)
    index = construct_kd_tree(db_points, 'manhattan')
    return index

# Streamlit
# left side: upload
with st.sidebar:
    st.header("Upload a 3D Model")
    uploaded_file = st.file_uploader("Upload a 3D OBJ file", type="obj")

index = set_ann_index()

    # right
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".obj") as temp_file:
        temp_file.write(uploaded_file.read())
        input_mesh_path = temp_file.name  # 保存临时文件路径

    K = st.sidebar.number_input("Number of results", min_value=1, max_value=2200)

    # match
    if st.sidebar.button("Match Model"):
        match_model(input_mesh_path, "descriptors_standardized.csv", "standardization_stats.csv", K=K)
    
    if st.sidebar.button("Fast Matching"):
        match_model(input_mesh_path, "descriptors_standardized.csv",
                    "standardization_stats.csv", K=K, fastMatch=True, ann_index=index)

else:
    st.write("Matching Results will be displayed here after uploading a model.")
