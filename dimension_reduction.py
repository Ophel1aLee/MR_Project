import argparse
import sklearn.manifold as sk
import pandas as pd
from matplotlib import pyplot as plt
from umap import umap_ as umap
from pynndescent import NNDescent
import pickle

def fit_tsne(data, perplexity, iterations):
    #tsne = sk.TSNE(perplexity=perplexity, n_iter=iterations, init='random', min_grad_norm=1e-8, learning_rate=150)
    model = umap.UMAP(n_neighbors=perplexity, n_epochs=iterations)
    newPoints = model.fit_transform(data)

    return (model, newPoints)

def ann(model: umap.UMAP, data, vector, K):
    point = model.transform([vector])
    index = NNDescent(data)

    results = index.query(point, K)

    return results

def plotResults(data):
    classes = data['class_name'].unique()
    colors = plt.get_cmap('Paired')

    class_names = []
    j = 0

    for i in range(len(classes)):
        points = data[data['class_name'] == classes[i]]
        plt.scatter(points['x'], points['y'], color=colors(i%12))
        class_names.append(classes[i])
        if (((i+1) % 12 == 0) or i == len(classes) - 1):
            plt.legend(class_names, loc='upper center', bbox_to_anchor=(0.5, -0.05),
                       ncol=3, fancybox=True, shadow=True)
            plt.savefig(f"figures/tsne_{j}.png", bbox_inches='tight')
            plt.show()
            class_names = []
            j += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--perplexity', type=int, default=30)
    parser.add_argument('-i', '--iterations', type=int, default=500)
    args = parser.parse_args()

    data = pd.read_csv("descriptors_standardized.csv")

    # Remove file path and class name from data
    vectors = data.drop(['class_name', 'file_name'], axis=1)

    model, reducedPoints = fit_tsne(vectors, args.perplexity, args.iterations)

    newData = pd.DataFrame(reducedPoints, columns=['x', 'y'])
    newData['class_name'] = data['class_name']
    newData['file_name'] = data['file_name']

    pickle.dump(model, open("umap_model.sav", 'wb'))

    newData.to_csv("fast_descriptors.csv")

    plotResults(newData)


