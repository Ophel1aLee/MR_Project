import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse

def plotPRCurve(data):
    plt.figure(figsize=(8, 6))
    plt.plot(data['Recall'], data['Precision'], linestyle='-', color='r')
    plt.axhline(0.0144, linestyle='--')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"figures/PRCurve.pdf")
    plt.show()


def plotPRperClass(data):
    ax = data.plot.bar(y='Precision', figsize=(14,6))
    ax.set_xticklabels(data.iloc[:,0])
    plt.tight_layout()
    plt.savefig(f"figures/PperClass.pdf")
    plt.show()
    ax = data.plot.bar(y='Recall', figsize=(14,6))
    ax.set_xticklabels(data.iloc[:, 0])
    plt.tight_layout()
    plt.savefig(f"figures/RperClass.pdf")
    plt.show()

def plotF1(data):
    prprod = data['Precision'] * data['Recall']
    prsum = data['Precision'] + data['Recall']
    F1s = 2 * (prprod / prsum)
    optimal = np.argmax(F1s)
    xrange = np.arange(2, 51)
    plt.figure(figsize=(8, 6))
    plt.plot(xrange, F1s, linestyle='-', color='r')
    plt.xlabel('query size (K)')
    plt.ylabel('F1')
    plt.title('F1 score per query size')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"figures/F1.pdf")
    print(f"Optimal K: {optimal+1}")
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', help='Which data to visualize ("prcurve", "prperclass" or "f1") (default=prcurve)', default='prcurve')
    args = parser.parse_args()

    if args.type == 'prcurve':
        data = pd.read_csv(f"pr_cache.csv")
        plotPRCurve(data)
    elif args.type == 'f1':
        data = pd.read_csv(f"pr_cache.csv")
        plotF1(data)
    elif args.type == 'prperclass':
        data = pd.read_csv(f"pr_per_class.csv")
        plotPRperClass(data)