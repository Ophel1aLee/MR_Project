from __future__ import annotations
import pandas as pd
import numpy as np
import open3d as o3d
from mesh_normalize import mesh_normalize_for_new
from mesh_descriptors import  three_d_property_descriptors, shape_property_descriptors
from ANN import ann
import trimesh
from stopwatch import Stopwatch
from mesh_resampling import mesh_resampling
from sklearn.metrics.pairwise import cosine_similarity


# The stopwatch is None by default, because it is only used during evaluation
def mesh_querying(model_file_name, csv_path, stats_path, K, stopwatch: Stopwatch | None = None):
    data = pd.read_csv(csv_path)

    # Process descriptors for the new model
    descriptors = process_new_model(model_file_name, stats_path)
    single_value_descriptors = descriptors[:6]
    histogram_descriptors = descriptors[6:]

    if stopwatch != None:
        stopwatch.start()

    # Extract features from the dataset
    single_value_features = data.iloc[:, 2:8].values
    histogram_features = [data.iloc[:, 8 + i * 100:8 + (i + 1) * 100].values for i in range(5)]

    # Compute cosine distances for single-value features
    # cosine_similarity outputs similarity; convert it to distance
    single_value_similarities = cosine_similarity(single_value_features, [single_value_descriptors]).flatten()
    single_value_distances = 1 - single_value_similarities

    # Calculate distance for each histogram feature group with special weight for the second histogram
    histogram_distances_list = []

    for i in range(5):
        # Store distances for the current histogram group
        histogram_distances = []
        for j in range(len(histogram_features[i])):
            # Calculate cosine similarity and convert to cosine distance
            similarity = cosine_similarity(
                [histogram_descriptors[i * 100:(i + 1) * 100]],  # Query histogram
                [histogram_features[i][j]]  # Dataset histogram
            )[0][0]
            # Convert similarity to distance
            distance = 1 - similarity
            histogram_distances.append(distance)

        # Convert distances to a numpy array for summation
        histogram_distances = np.array(histogram_distances)

        histogram_distances_list.append(histogram_distances)

    # Compute total distances with the adjusted contribution for the second histogram
    total_distances = single_value_distances + sum(histogram_distances_list)

    # Find the indices of the K most similar models
    closest_indices = np.argsort(total_distances)[:K]
    closest_models = data.iloc[closest_indices][['class_name', 'file_name']].values
    closest_distances = total_distances[closest_indices]
    if stopwatch != None:
        stopwatch.stop()
        stopwatch.record_time()
    return [model[0] for model in closest_models], list(zip(closest_models, closest_distances))


def process_new_model(input_mesh_path, stats_path):
    resampled_mesh = mesh_resampling(input_mesh_path, 5000, 200, 20)
    resampled_mesh.save_current_mesh("temp.obj")

    resampled_mesh = o3d.io.read_triangle_mesh("temp.obj")
    normalized_mesh = mesh_normalize_for_new(resampled_mesh)
    normalized_mesh.compute_vertex_normals()
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


def fast_query(input_mesh_path, stats_path, descriptors_path, ann_index, K, stopwatch: Stopwatch | None = None):
    descriptors = process_new_model(input_mesh_path, stats_path)
    db_descriptors = pd.read_csv(descriptors_path)
    if stopwatch != None:
        stopwatch.start()
    indices, distances = ann(ann_index, descriptors, K)
    closest_models = db_descriptors.iloc[indices[0,:]][['class_name', 'file_name']].values
    closest_distances = distances[0,:]
    if stopwatch != None:
        stopwatch.stop()
        stopwatch.record_time()

    return [model[0] for model in closest_models], list(zip(closest_models, closest_distances))