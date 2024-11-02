import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


def mesh_querying(model_file_name, csv_path):
    data = pd.read_csv(csv_path)

    model_row = data[data['file_name'] == model_file_name]
    if model_row.empty:
        return f"Model file '{model_file_name}' not found in the CSV file."
    class_name = model_row['class_name'].values[0]
    model_features = model_row.iloc[:, 2:].values

    # Filter the dataset to contain only models from the same class
    class_data = data[data['class_name'] == class_name]
    # Remove the target model itself from the dataset for comparison
    class_data_filtered = class_data[class_data['file_name'] != model_file_name]

    # Extract features for all models in the same class (excluding the target model)
    class_features = class_data_filtered.iloc[:, 2:].values

    # Compute Euclidean distances between the target model and all others in the same class
    distances = euclidean_distances(model_features, class_features)[0]

    # Get the indices of the four closest models (smallest distances)
    closest_indices = np.argsort(distances)[:4]

    # Retrieve file names of the closest models
    closest_models = class_data_filtered.iloc[closest_indices]['file_name'].values
    closest_distances = distances[closest_indices]

    return class_name, list(zip(closest_models, closest_distances))


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