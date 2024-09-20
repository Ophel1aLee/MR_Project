import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("mesh_analysis_cache.csv")

number_verts = data['vertices']
number_faces = data['triangles']
classes = data['class']

avg_verts = number_verts.sum() / number_verts.size
avg_faces = number_faces.sum() / number_faces.size

print("Average vertices: " + str(avg_verts))
print("Average faces: " + str(avg_faces))

number_verts.plot.hist(bins=100)

plt.show()

number_faces.plot.hist(bins=100)

plt.show()

classes.value_counts().plot.bar()

plt.show()