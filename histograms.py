import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

data = pd.read_csv("mesh_analysis_cache.csv")

number_verts = data['vertices']
number_faces = data['triangles']
classes = data['class']

avg_verts = number_verts.sum() / number_verts.size
avg_faces = number_faces.sum() / number_faces.size

print("Average vertices: " + str(avg_verts))
print("Average faces: " + str(avg_faces))

volumes = []
toolarge = 0

for i in range(len(classes)):
    volume = abs(data['minx'][i] - data['maxx'][i]) * \
        abs(data['miny'][i] - data['maxy'][i]) * \
        abs(data['minz'][i] - data['maxz'][i])
    if volume < 10:
        volumes.append(volume)
    else:
        toolarge += 1

number_verts.plot.hist(bins=100)
plt.show()
number_faces.plot.hist(bins=100)
plt.show()
classes.value_counts().plot.bar()
plt.show()
print(np.average(volumes))
print(np.var(volumes))
print(f"Too large: {toolarge}")
bins = np.arange(min(volumes), max(volumes) + .1, .1)
plt.hist(volumes, bins=bins)
plt.show()