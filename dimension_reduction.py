import argparse
import pandas as pd
from matplotlib import pyplot as plt
from umap import umap_ as umap

# Use the feature vectors to train a UMAP model, and return the transformed data points.
def fit_umap(data, perplexity, iterations):
    model = umap.UMAP(n_neighbors=perplexity, n_epochs=iterations)
    newPoints = model.fit_transform(data)

    return (model, newPoints)

def plotResults(data):
    classes = data['class_name'].unique()
    colors = plt.get_cmap('Paired') # This colormap contains 12 distinct colors, which is exactly the number we need

    class_names = []
    j = 0

    for i in range(len(classes)):
        points = data[data['class_name'] == classes[i]]
        plt.scatter(points['x'], points['y'], color=colors(i%12))
        class_names.append(classes[i])
        if (((i+1) % 12 == 0) or i == len(classes) - 1): # Return a scatterplot after each set of 12 classes
            plt.legend(class_names, loc='upper center', bbox_to_anchor=(0.5, -0.05),
                       ncol=3, fancybox=True, shadow=True)
            plt.savefig(f"figures/tsne_{j}.png", bbox_inches='tight')
            plt.show()
            class_names = []
            j += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--perplexity', help='The radius from each point within which UMAP tries to retain distances (default=35)', type=int, default=35)
    parser.add_argument('-i', '--iterations', help='The maximum number of iterations (default=1500)', type=int, default=1500)
    args = parser.parse_args()

    data = pd.read_csv("descriptors_standardized.csv")

    # Remove file path and class name from data
    vectors = data.drop(['class_name', 'file_name'], axis=1)

    model, reducedPoints = fit_umap(vectors, args.perplexity, args.iterations)

    newData = pd.DataFrame(reducedPoints, columns=['x', 'y'])

    # Add class and file name back to the transformed points
    newData['class_name'] = data['class_name']
    newData['file_name'] = data['file_name']

    plotResults(newData)