# Do the resampling
#from mesh_resampling import resample_database
#resample_database("ShapeDatabase", "ShapeDatabase_Resampled", 5000, 200)

# Do the normalizing
#from mesh_normalize import normalize_database
#normalize_database("ShapeDatabase_Resampled", "ShapeDatabase_Normalized")

# Calculate the descriptors for the whole database
from mesh_descriptors import calculate_descriptor_for_the_database
calculate_descriptor_for_the_database("ShapeDatabase_Normalized", 150000, 100)