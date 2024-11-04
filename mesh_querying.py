import open3d.cpu.pybind.io
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import open3d as o3d
from mesh_resampling import mesh_resampling
from mesh_normalize import mesh_normalize_for_new
from mesh_descriptors import  three_d_property_descriptors, shape_property_descriptors
from ANN import ann
import trimesh
import pymeshlab as pml
from sklearn.preprocessing import StandardScaler
import pickle


def mesh_querying(model_file_name, csv_path, stats_path, K):
    data = pd.read_csv(csv_path)

    descriptors = process_new_model(model_file_name, stats_path)
    single_value_descriptors = descriptors[:6]  # Assuming columns 3 to 8 are single-value features
    histogram_descriptors = descriptors[6:]

    single_value_features = data.iloc[:, 2:8].values
    histogram_features = [data.iloc[:, 8 + i * 100:8 + (i + 1) * 100].values for i in range(5)]

    # Compute Euclidean distances for single-value features
    single_value_distances = euclidean_distances([single_value_descriptors], single_value_features)[0]
    standardized_histogram_distances = []
    scaler = StandardScaler()
    for i in range(5):
        histogram_distances = \
        euclidean_distances([histogram_descriptors[i * 100:(i + 1) * 100]], histogram_features[i])[0]
        standardized_histogram_distances.append(scaler.fit_transform(histogram_distances.reshape(-1, 1)).flatten())

    #distances = euclidean_distances([descriptors], all_features)[0]
    total_distances = single_value_distances + sum(standardized_histogram_distances)
    #closest_indices = np.argsort(distances)[:K]
    closest_indices = np.argsort(total_distances)[:K]
    closest_models = data.iloc[closest_indices][['class_name', 'file_name']].values
    closest_distances = total_distances[closest_indices]
    return [model[0] for model in closest_models], list(zip(closest_models, closest_distances))

    # else:
    #     # Extract the row corresponding to the target model
    #     model_row = data[data['file_name'] == model_file_name]
    #     class_name = model_row['class_name'].values[0]
    #     model_features = model_row.iloc[:, 2:].values

    #     # Filter the dataset to contain only models from the same class
    #     class_data = data[data['class_name'] == class_name]
    #     # Remove the target model itself from the dataset for comparison
    #     class_data_filtered = class_data[class_data['file_name'] != model_file_name]

    #     # Extract features for all models in the same class (excluding the target model)
    #     class_features = class_data_filtered.iloc[:, 2:].values

    #     # Compute Euclidean distances between the target model and all others in the same class
    #     distances = euclidean_distances(model_features, class_features)[0]

    #     # Get the indices of the four closest models (smallest distances)
    #     closest_indices = np.argsort(distances)[:4]

    #     # Retrieve file names of the closest models
    #     closest_models = class_data_filtered.loc[class_data_filtered.index[closest_indices], ['class_name', 'file_name']].values
    #     closest_distances = distances[closest_indices]

    #     return class_name, list(zip(closest_models, closest_distances))


def mesh_querying_global(model_file_name, csv_path):
    data = pd.read_csv(csv_path)

    model_row = data[data['file_name'] == model_file_name]
    if model_row.empty:
        return f"Model file '{model_file_name}' not found in the CSV file."
    model_features = model_row.iloc[:, 2:].values

    # Remove the target model itself from the dataset for comparison
    data_filtered = data[data['file_name'] != model_file_name]

    # Extract features for all models in the dataset (excluding the target model)
    all_features = data_filtered.iloc[:, 2:].values

    # Compute Euclidean distances between the target model and all others in the dataset
    distances = euclidean_distances(model_features, all_features)[0]

    # Get the indices of the four closest models (smallest distances)
    closest_indices = np.argsort(distances)[:4]

    # Retrieve file names, class names, and distances of the closest models
    closest_models = data_filtered.iloc[closest_indices][['class_name', 'file_name']].values
    closest_distances = distances[closest_indices]

    return list(zip(closest_models, closest_distances))


def process_new_model(input_mesh_path, stats_path):
    mesh = pml.MeshSet()
    mesh.load_new_mesh(input_mesh_path)
    resampled_mesh = mesh_resampling(mesh, 5000, 200, 20)
    resampled_mesh.save_current_mesh("temp.obj")

    resampled_mesh = o3d.io.read_triangle_mesh("temp.obj")
    normalized_mesh = mesh_normalize_for_new(resampled_mesh)
    o3d.io.write_triangle_mesh("temp.obj", normalized_mesh, write_vertex_normals=True)

    normalized_mesh = trimesh.load("temp.obj")
    surface_area, compactness, rectangularity, diameter, convexity, eccentricity = three_d_property_descriptors(normalized_mesh)
    A3_hist, D1_hist, D2_hist, D3_hist, D4_hist = shape_property_descriptors(normalized_mesh, 150000, 100)

    # Standardize single-value descriptors
    stats = pd.read_csv(stats_path)
    single_value_descriptors = [surface_area, compactness, rectangularity, diameter, convexity, eccentricity]
    standardized_descriptors = []
    for i, descriptor in enumerate(single_value_descriptors):
        mean = stats.loc[i, 'mean']
        std = stats.loc[i, 'std']
        standardized_value = (descriptor - mean) / std
        standardized_descriptors.append(standardized_value)

    # Normalize histogram descriptors
    A3_hist = A3_hist / np.sum(A3_hist)
    D1_hist = D1_hist / np.sum(D1_hist)
    D2_hist = D2_hist / np.sum(D2_hist)
    D3_hist = D3_hist / np.sum(D3_hist)
    D4_hist = D4_hist / np.sum(D4_hist)

    descriptors = [
        *standardized_descriptors,
        *A3_hist, *D1_hist, *D2_hist, *D3_hist, *D4_hist
    ]

    return descriptors

def fast_query(input_mesh_path, stats_path, descriptors_path, K):
    descriptors = process_new_model(input_mesh_path, stats_path)
    #umap_model = pickle.load((open('umap_model.sav', 'rb')))
    db_descriptors = pd.read_csv(descriptors_path)
    db_points = db_descriptors.drop(['class_name', 'file_name'], axis=1)

    indices, distances = ann(db_points.to_numpy(), descriptors, K)

    print(indices)
    print(indices[0,:])

    closest_models = db_descriptors.iloc[indices[0,:]][['class_name', 'file_name']].values
    closest_distances = distances[0,:]

    return [model[0] for model in closest_models], list(zip(closest_models, closest_distances))


