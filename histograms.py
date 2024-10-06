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
    plt.savefig(f"figures/{filename}_{version}.png")
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
    print("Analyzing translation")

def analyze_rotation():
    print("Analyzing rotation")

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
    print("Analyzing flip direction")

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