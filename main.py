# Analyze the database
from mesh_analyze import analyze_mesh_in_folder
analyze_mesh_in_folder("ShapeDatabase")

# Do the resampling
from mesh_resampling import resample_database
resample_database("ShapeDatabase", "ShapeDatabase_Resampled", 5000, 200)

# Do the normalizing
from mesh_normalize import normalize_database
normalize_database("ShapeDatabase_Resampled", "ShapeDatabase_Normalized")

# Calculate the descriptors for the whole database
from mesh_descriptors import calculate_descriptor_for_the_database
calculate_descriptor_for_the_database("ShapeDatabase_Normalized", 150000, 100)

# Extract shape property descriptors histogram datas and plot from the CSV file
from visualize_shape_histogram import plot_histograms_from_csv
plot_histograms_from_csv('descriptors.csv', 'D00138.obj')

# Normalizing the histograms of the shape descriptors
from descriptors_processing import histogram_normalizing
histogram_normalizing("descriptors.csv", "descriptors_normalized.csv", 100)

# Plot histograms for class
from descriptors_processing import plot_histograms_for_class
plot_histograms_for_class("descriptors_normalized.csv", "Cup", 100)

# Normalizing the single descriptors of the 3D descriptors
from descriptors_processing import single_value_normalizing
single_value_normalizing("descriptors_normalized.csv", "descriptors_standardized.csv")