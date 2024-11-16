import random
import math
import numpy as np
import trimesh
from scipy.spatial.distance import pdist
import pandas as pd
import os


def signed_volume_of_triangle(p1, p2, p3):
    v321 = p3[0] * p2[1] * p1[2]
    v231 = p2[0] * p3[1] * p1[2]
    v312 = p3[0] * p1[1] * p2[2]
    v132 = p1[0] * p3[1] * p2[2]
    v213 = p2[0] * p1[1] * p3[2]
    v123 = p1[0] * p2[1] * p3[2]
    return (1.0 / 6.0) * (-v321 + v231 + v312 - v132 - v213 + v123)


def sum_volume_of_mesh(triangles):
    vols = [signed_volume_of_triangle(t[0], t[1], t[2]) for t in triangles]
    return abs(sum(vols))


def calculate_volume(mesh):
    triangles = []
    # 提取顶点和面信息
    vertices = [tuple(vertex) for vertex in mesh.vertices]
    for face in mesh.faces:
        # 假设面是三角化的
        idx1, idx2, idx3 = face
        triangles.append((vertices[idx1], vertices[idx2], vertices[idx3]))
    volume = sum_volume_of_mesh(triangles)
    return volume


# 3D property descriptors
def three_d_property_descriptors(mesh):
    surface_area = mesh.area

    mesh_volume = calculate_volume(mesh)
    compactness = (surface_area ** 3) / (36 * np.pi * mesh_volume ** 2)

    obb = mesh.bounding_box_oriented
    obb_volume = obb.volume
    rectangularity = mesh_volume / obb_volume

    diameter = np.max(pdist(mesh.vertices))

    convex_hull = mesh.convex_hull
    hull_volume = convex_hull.volume
    convexity = mesh_volume / hull_volume

    covariance_matrix = np.cov(mesh.vertices, rowvar=False)
    eigenvalues, _ = np.linalg.eig(covariance_matrix)
    eccentricity = max(eigenvalues) / min(eigenvalues)

    return (surface_area, compactness, rectangularity, diameter, convexity, eccentricity)


# A3
def calculate_A3(vertices, n):
    N = len(vertices)

    # Compute number of samples along each of the three dimensions
    k = int(math.pow(n, 1.0 / 3.0))
    # Store calculated angles
    angles = []

    for i in range(k):
        vi = random.randint(0, N - 1)

        for j in range(k):
            vj = random.randint(0, N - 1)
            if vj == vi:
                continue

            for l in range(k):
                vl = random.randint(0, N - 1)
                if vl == vi or vl == vj:
                    continue

                # Get coordinates of three vertices
                A = np.array(vertices[vi])
                B = np.array(vertices[vj])
                C = np.array(vertices[vl])

                # Calculate vectors AB and AC
                AB = B - A
                AC = C - A

                # Skip the situation where the vector is 0 to avoid dividing by 0
                if np.linalg.norm(AB) == 0 or np.linalg.norm(AC) == 0:
                    continue

                # Calculate the angle
                cos_theta = np.dot(AB, AC) / (np.linalg.norm(AB) * np.linalg.norm(AC))
                # Make sure the cosine is between [-1, 1]
                angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
                # Convert radians to angles and store them
                angles.append(np.degrees(angle))
    return angles


# D1
def calculate_D1(vertices):
    barycenter = np.mean(vertices, axis=0)
    distances = []

    # Traverse all vertices and calculate their distance to the centroid
    for vertex in vertices:
        distance = np.linalg.norm(np.array(vertex) - barycenter)
        distances.append(distance)
    return distances


# D2
def calculate_D2(vertices, n):
    N = len(vertices)
    k = int(math.pow(n, 1.0 / 2.0))
    distances = []

    for i in range(k):
        vi = random.randint(0, N - 1)

        for j in range(k):
            vj = random.randint(0, N - 1)
            if vj == vi:
                continue

            A = np.array(vertices[vi])
            B = np.array(vertices[vj])

            # Calculate Euclidean distance
            distance = np.linalg.norm(B - A)
            distances.append(distance)
    return distances


# D3
def calculate_D3(vertices, n):
    N = len(vertices)
    k = int(math.pow(n, 1.0 / 3.0))
    areas = []

    for i in range(k):
        vi = random.randint(0, N - 1)

        for j in range(k):
            vj = random.randint(0, N - 1)
            if vj == vi:
                continue

            for l in range(k):
                vl = random.randint(0, N - 1)
                if vl == vi or vl == vj:
                    continue

                A = np.array(vertices[vi])
                B = np.array(vertices[vj])
                C = np.array(vertices[vl])

                # Calculate the cross product of AB and AC
                AB = B - A
                AC = C - A
                cross_product = np.cross(AB, AC)

                # The area of a triangle is the length of the cross product divided by 2
                area = np.linalg.norm(cross_product) / 2.0
                areas.append(area)
    return areas


# D4
def calculate_D4(vertices, n):
    N = len(vertices)
    k = int(math.pow(n, 1.0 / 4.0))
    volumes_cube_root = []

    for i in range(k):
        vi = random.randint(0, N - 1)

        for j in range(k):
            vj = random.randint(0, N - 1)
            if vj == vi:
                continue

            for l in range(k):
                vl = random.randint(0, N - 1)
                if vl == vi or vl == vj:
                    continue

                for m in range(k):
                    vm = random.randint(0, N - 1)
                    if vm == vi or vm == vj or vm == vl:
                        continue

                    A = np.array(vertices[vi])
                    B = np.array(vertices[vj])
                    C = np.array(vertices[vl])
                    D = np.array(vertices[vm])

                    volume = np.abs(np.dot(A - D, np.cross(B - D, C - D))) / 6.0

                    # Calculate the cube root of the volume
                    cube_root_volume = volume ** (1 / 3)
                    volumes_cube_root.append(cube_root_volume)
    return volumes_cube_root


# Shape property descriptors
def shape_property_descriptors(mesh, n_samples, histogram_bins):
    vertices = np.asarray(mesh.vertices)
    A3 = calculate_A3(vertices, n_samples)
    D1 = calculate_D1(vertices)
    D2 = calculate_D2(vertices, n_samples)
    D3 = calculate_D3(vertices, n_samples)
    D4 = calculate_D4(vertices, n_samples)

    A3_hist, _ = np.histogram(A3, bins=histogram_bins, density=False)
    D1_hist, _ = np.histogram(D1, bins=histogram_bins, density=False)
    D2_hist, _ = np.histogram(D2, bins=histogram_bins, density=False)
    D3_hist, _ = np.histogram(D3, bins=histogram_bins, density=False)
    D4_hist, _ = np.histogram(D4, bins=histogram_bins, density=False)

    return (A3_hist, D1_hist, D2_hist, D3_hist, D4_hist)


# Calculate descriptors for the database
def calculate_descriptors(meshes, n_samples, histogram_bins):
    data = []
    single_value_features = []

    for mesh, class_name, file_name in meshes:
        # Calculate 3D property descriptors
        surface_area, compactness, rectangularity, diameter, convexity, eccentricity = three_d_property_descriptors(mesh)

        # Calculate shape property descriptors
        A3_hist, D1_hist, D2_hist, D3_hist, D4_hist = shape_property_descriptors(mesh, n_samples, histogram_bins)

        # Turn the datas in to a dictionary
        result = {
            'class_name': class_name,
            'file_name': file_name,
            'surface_area': surface_area,
            'compactness': compactness,
            'rectangularity': rectangularity,
            'diameter': diameter,
            'convexity': convexity,
            'eccentricity': eccentricity,
        }
        # Save single valued features for subsequent standardization
        single_value_features.append([surface_area, compactness, rectangularity, diameter, convexity, eccentricity])

        # Add histograms
        for i in range(histogram_bins):
            result[f'A3_hist_bin_{i + 1}'] = A3_hist[i]
        for i in range(histogram_bins):
            result[f'D1_hist_bin_{i + 1}'] = D1_hist[i]
        for i in range(histogram_bins):
            result[f'D2_hist_bin_{i + 1}'] = D2_hist[i]
        for i in range(histogram_bins):
            result[f'D3_hist_bin_{i + 1}'] = D3_hist[i]
        for i in range(histogram_bins):
            result[f'D4_hist_bin_{i + 1}'] = D4_hist[i]

        # Save the data
        data.append(result)

    # Add results
    for i, result in enumerate(data):
        result['surface_area'] = single_value_features[i][0]
        result['compactness'] = single_value_features[i][1]
        result['rectangularity'] = single_value_features[i][2]
        result['diameter'] = single_value_features[i][3]
        result['convexity'] = single_value_features[i][4]
        result['eccentricity'] = single_value_features[i][5]

    # Return descriptor data and mean standard deviation
    return data


def save_descriptors_to_csv(data, output_file):
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"Descriptors saved to {output_file}")


# Save the mean and the standard deviation
def save_stats_to_csv(mean_values, std_values, output_file):
    stats = pd.DataFrame({
        'feature': ['surface_area', 'compactness', 'rectangularity', 'diameter', 'convexity', 'eccentricity'],
        'mean': mean_values,
        'std': std_values
    })
    stats.to_csv(output_file, index=False)
    print(f"Mean and standard deviation saved to {output_file}")


# Load the database
def calculate_descriptor_for_the_database(input_folder, n_samples, histogram_bins):
    meshes = []
    for class_name in os.listdir(input_folder):
        class_dir = os.path.join(input_folder, class_name)
        if os.path.isdir(class_dir):
            for file_name in os.listdir(class_dir):
                file_path = os.path.join(class_dir, file_name)
                mesh = trimesh.load(file_path)
                meshes.append((mesh, class_name, file_name))

    descriptors = calculate_descriptors(meshes, n_samples, histogram_bins)
    save_descriptors_to_csv(descriptors, 'descriptors.csv')
