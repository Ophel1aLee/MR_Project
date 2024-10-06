import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import math
import argparse

def nbins(n) -> int:
    return int(math.sqrt(len(n)))

def printStats(variable, data: list[float | int]):
    print(variable)
    print("\t Mean: " + str(np.average(data)))
    print("\t SD: " + str(math.sqrt(np.var(data))))

def makeHist(title: str, xlabel: str, data: list[float | int]):
    plt.hist(data, bins=nbins(data), color="C0")
    plt.title(title)
    plt.xlabel(xlabel)
    filename = title.lower().replace(" ", "_")
    plt.savefig(f"figures/{filename}_{version}.pdf")
    plt.show()

def analyze_sample_count():
    print("== Analyzing sample count ==")
    number_verts = cache_data['vertices']
    number_faces = cache_data['triangles']

    printStats("Vertices", number_verts)
    printStats("Faces", number_faces)
    makeHist("Number of Vertices", "Vertex Count", number_verts)
    makeHist("Number of Faces", "Face Count", number_faces)

def analyze_edge_variance():
    print("== Analyzing edge variance ==")
    edge_var = cache_data['edge_var']
    edge_var_trunc = []

    for e in edge_var:
        if e < 1:
            edge_var_trunc.append(e)
    
    printStats("Edge Variance", edge_var_trunc)
    makeHist("Edge Length Variance", "Variances", edge_var_trunc)

def analyze_translation():
    print("== Analyzing translation ==")
    paths = cache_data['file']
    distances = []

    for p in paths:
        mesh = o3d.io.read_triangle_mesh(p)
        centroid = mesh.get_center()
        dist = np.linalg.norm(centroid - np.array((0,0,0)))
        if dist < 20:
            distances.append(dist)
        else:
            print(dist)
    
    printStats("Distance to World Origin", distances)
    makeHist("Distance to World Origin", "Distances", distances)

def analyze_rotation():
    print("== Analyzing rotation ==")
    paths = cache_data['file']
    dotxs = []
    dotys = []
    dotzs = []

    invalid = 0

    for p in paths:
        mesh = o3d.io.read_triangle_mesh(p)
        covariance = np.cov(np.array(mesh.vertices).T)
        
        try:
            _, eigenvectors = np.linalg.eigh(covariance)
            l1 = eigenvectors[:, 0] # x
            l2 = eigenvectors[:, 1] # y
            l3 = eigenvectors[:, 2] # z

            dotxs.append(np.dot(l1, np.array((1, 0, 0))))
            dotys.append(np.dot(l2, np.array((0, 1, 0))))
            dotzs.append(np.dot(l3, np.array((0, 0, 1))))
        except:
            invalid += 1
    
    printStats("Dot l1 x", dotxs)
    printStats("Dot l2 y", dotys)
    printStats("Dot l3 z", dotzs)
    print("\t Invalid: " + str(invalid))

    makeHist("Dot Product L1", "L1 . x", dotxs)
    makeHist("Dot Product L2", "L2 . y", dotys)
    makeHist("Dot Product L3", "L3 . z", dotzs)

def analyze_scale():
    print("== Analyzing scale ==")
    toolarge = 0

    max_sides = []

    for i in range(len(cache_data['minx'])):
        max_side = max([abs(cache_data['minx'][i] - cache_data['maxx'][i]), \
                abs(cache_data['miny'][i] - cache_data['maxy'][i]), \
                abs(cache_data['minz'][i] - cache_data['maxz'][i])])
        if max_side < 20:
            max_sides.append(max_side)
        else:
            toolarge += 1

    printStats("AABB Length", max_sides)
    print(f"\t Discarded: {toolarge}")
    makeHist("AABB Length", "L_max", max_sides)


def analyze_flip_direction():
    print("== Analyzing flip direction ==")
    paths = cache_data['file']
    xspos = xsneg = 0
    yspos = ysneg = 0
    zspos = zsneg = 0

    invalid = 0

    for p in paths:
        mesh = o3d.io.read_triangle_mesh(p)
        vertices = np.asarray(mesh.vertices)
        mass_center = np.mean(vertices, axis=0)

        if mass_center[0] < 0:  # x
            xsneg += 1
        elif mass_center[0] >= 0:
            xspos += 1
        if mass_center[1] < 0:  # x
            ysneg += 1
        elif mass_center[1] >= 0:
            yspos += 1
        if mass_center[2] < 0:  # x
            zsneg += 1
        elif mass_center[2] >= 0:
            zspos += 1

    fig, ax = plt.subplots(layout='constrained')
    rects = ax.bar(np.array((-1, 1)), np.array((xsneg, xspos)), 0.25, label="x")
    ax.bar_label(rects, padding=3)
    rects = ax.bar(np.array((-1, 1)) + 0.25, np.array((ysneg, yspos)), 0.25, label="y")
    ax.bar_label(rects, padding=3)
    rects = ax.bar(np.array((-1, 1)) + 0.5, np.array((zsneg, zspos)), 0.25, label="z")
    ax.bar_label(rects, padding=3)
    ax.set_xticks(np.array((-1, 1)) + 0.25, np.array((-1, 1)))
    ax.legend(loc='upper left', ncols=3)
    ax.set_ylabel("Frequency")
    ax.set_xlabel("Direction")
    ax.set_title("Direction of Mass")
    plt.savefig(f"figures/direction_of_mass_{version}.png")
    plt.show()

def main(values: list[str]):
    all_values = ["sample_count", "edge_variance", "translation", "rotation", "scale", "flip"]
    all_functions = [analyze_sample_count, analyze_edge_variance, analyze_translation, analyze_rotation, analyze_scale, analyze_flip_direction]

    operations = zip(all_values, all_functions)

    print("======================")

    for (v, f) in operations:
        if v in values:
            f()
            print("======================")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', help='Version of the database to be analyzed (original, fixed, resampled, normalized) (default=original)', default='original')
    parser.add_argument('--values', help='Comma-separated list of values to be analyzed (sample_count, edge_variance, translation, rotation, scale, flip)(default=all)',
                        default='sample_count, edge_variance, translation, rotation, scale, flip')
    args = parser.parse_args()

    version = args.version
    cache_file = f"mesh_analysis_cache_{args.version}.csv"
    cache_data = pd.read_csv(cache_file)

    # remove all whitespace in the string, and split it on comma chars
    values = "".join(args.values.split()).split(',')

    main(values)