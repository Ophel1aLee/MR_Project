import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import math


def nbins(n) -> int:
    return int(math.sqrt(len(n)))

data = pd.read_csv("mesh_analysis_cache.csv")

number_verts = data['vertices']
number_faces = data['triangles']
classes = data['class']
edge_var = data['edge_var']

avg_verts = number_verts.sum() / number_verts.size
avg_faces = number_faces.sum() / number_faces.size
avg_edge_var = edge_var.sum() / edge_var.size

print("Average vertices: " + str(avg_verts))
print("Std dev: " + str(math.sqrt(np.var(number_verts))))
print("Average faces: " + str(avg_faces))
print("Std dev: " + str(math.sqrt(np.var(number_faces))))
print("Average Edge var: " + str(avg_edge_var))
print("SD: " + str(math.sqrt(np.var(edge_var))))

volumes = []
toolarge = []

max_sides = []

for i in range(len(classes)):
    max_side = max([abs(data['minx'][i] - data['maxx'][i]), \
            abs(data['miny'][i] - data['maxy'][i]), \
            abs(data['minz'][i] - data['maxz'][i])])
    if max_side < 20:
        max_sides.append(max_side)
    else:
        toolarge.append(max_side)

edge_var_trunc = []

for e in edge_var:
    if e < 1:
        edge_var_trunc.append(e)

number_verts.plot.hist(bins=nbins(number_verts))
plt.show()
number_faces.plot.hist(bins=nbins(number_faces))
plt.show()
#bins = np.arange(min(edge_var), max(edge_var) + .01, .01)
plt.hist(edge_var_trunc, bins=nbins(edge_var_trunc))
plt.show()
classes.value_counts().plot.bar()
plt.show()
print(np.average(max_sides))
print(np.var(max_sides))
print(f"Too large: {len(toolarge)}")
print(toolarge)
bins = np.arange(min(max_sides), max(max_sides) + .1, .1)
plt.hist(max_sides, bins=nbins(max_sides))
plt.show()

