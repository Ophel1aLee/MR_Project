from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import math
import argparse
from mesh_normalize import triangleCenter, sign


def nbins(n) -> int:
    return int(math.sqrt(len(n)))


def printStats(variable, data: list[float | int]):
    print(variable)
    print("\t Mean: " + str(np.average(data)))
    print("\t SD: " + str(math.sqrt(np.var(data))))


def makeHist(title: str, xlabel: str, data: list[float | int], bins: list[float] | None = None):
    b = nbins(data) if bins is None else bins
    print("Number of bins: " + str(b))
    plt.hist(data, bins=b, color="C0")
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
    makeHist("Number of Vertices", "Vertex Count",
             number_verts, np.arange(0, 50000, 1000, float))
    makeHist("Number of Faces", "Face Count",
             number_faces, np.arange(0, 50000, 1000, float))


def analyze_edge_variance():
    print("== Analyzing edge variance ==")
    edge_var = cache_data['edge_var']
    edge_var_trunc = []

    for e in edge_var:
        if e < 1:
            edge_var_trunc.append(e)
    
    printStats("Edge Variance", edge_var_trunc)
    makeHist("Edge Length Variance", "Variances",
             edge_var_trunc, np.arange(0, 1.02, 0.02, float))


def analyze_translation():
    print("== Analyzing translation ==")
    paths = cache_data['file']
    distances = []

    for p in paths:
        mesh = o3d.io.read_triangle_mesh(p, enable_post_processing=True)
        mesh = mesh.remove_duplicated_vertices()
        centroid = mesh.get_center()
        dist = np.linalg.norm(centroid - np.array((0,0,0)))
        if dist < 20:
            distances.append(dist)
        else:
            print(dist)
    
    printStats("Distance to World Origin", distances)
    makeHist("Distance to World Origin", "Distances", distances, np.arange(0, 5, 0.1, float))


def analyze_rotation():
    print("== Analyzing rotation ==")
    paths = cache_data['file']
    dotxs = []
    dotys = []
    dotzs = []

    invalid = 0
    incorrect = 0
    total = len(paths)

    for p in paths:
        mesh = o3d.io.read_triangle_mesh(p, enable_post_processing=True)
        mesh = mesh.remove_duplicated_vertices()
        covariance = np.cov(np.asarray(mesh.vertices).T)

        
        try:
            eigenvalues, eigenvectors = np.linalg.eig(covariance)
            eigencombined = [(eigenvalues[i], eigenvectors[:, i]) for i in range(3)]
            eigencombined.sort(key=lambda x: x[0], reverse=True)
            eigenvectors = [item[1] for item in eigencombined]
            eigenvalues = [item[0] for item in eigencombined]

            eigenvectors.pop(2)
            eigenvectors.append(np.cross(eigenvectors[0], eigenvectors[1]))

            ev = eigenvectors

            l1 = ev[0]
            l2 = ev[1]
            l3 = np.cross(l1, l2)

            dotx = abs(np.dot(l1, np.array((1, 0, 0))))
            doty = abs(np.dot(l2, np.array((0, 1, 0))))
            dotz = abs(np.dot(l3, np.array((0, 0, 1))))

            dotxs.append(dotx)
            dotys.append(doty)
            dotzs.append(dotz)

            if dotx < 0.9 or doty < 0.9 or dotz < 0.9:
                incorrect += 1
                print(p)

        except:
            invalid += 1
    
    printStats("Dot e_1 x", dotxs)
    printStats("Dot e_2 y", dotys)
    printStats("Dot e_3 z", dotzs)
    print("\t Invalid: " + str(invalid))
    print(f"\t Incorrect: {incorrect}/{total}")

    makeHist("Dot Product e_1", "e_1 . x", dotxs, np.arange(0, 1.02, 0.02, float))
    makeHist("Dot Product e_2", "e_2 . y", dotys, np.arange(0, 1.02, 0.02, float))
    makeHist("Dot Product e_3", "e_3 . z", dotzs, np.arange(0, 1.02, 0.02, float))


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
    makeHist("AABB Length", "L_max", max_sides, np.arange(0, 5, 0.1, float))


def analyze_flip_direction():
    print("== Analyzing flip direction ==")
    paths = cache_data['file']
    xspos = xsneg = 0
    yspos = ysneg = 0
    zspos = zsneg = 0

    total = 0

    invalid = 0

    for p in paths:
        mesh = o3d.io.read_triangle_mesh(p, enable_post_processing=True)
        mesh = mesh.remove_duplicated_vertices()
        vertices = np.copy(np.asarray(mesh.vertices))

        fx = fy = fz = 0

        for a, b, c in np.asarray(mesh.triangles):
            tricenter = triangleCenter(vertices[a], vertices[b], vertices[c])
            fx += sign(tricenter[0]) * (tricenter[0] * tricenter[0])
            fy += sign(tricenter[1]) * (tricenter[1] * tricenter[1])
            fz += sign(tricenter[2]) * (tricenter[2] * tricenter[2])
        
        total += 1
        
        if sign(fx) == -1:
            xsneg += 1
        if sign(fy) == -1:
            ysneg += 1
        if sign(fz) == -1:
            zsneg += 1
        
    xspos = total - xsneg
    yspos = total - ysneg
    zspos = total - zsneg

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
    plt.savefig(f"figures/direction_of_mass_{version}.pdf")
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