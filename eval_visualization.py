import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse

def plotPRCurve(data, version):
    plt.figure(figsize=(8, 6))
    plt.plot(data['Recall'], data['Precision'], linestyle='-', color='r')
    plt.axhline(0.0144, linestyle='--')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"figures/PRCurve_{version}.pdf")
    plt.show()


def plotPRperClass(data, version):
    ax = data.plot.bar(y='Precision', figsize=(14,6))
    ax.set_xticklabels(data.iloc[:,0])
    plt.tight_layout()
    plt.savefig(f"figures/PperClass_{version}.pdf")
    plt.show()
    ax = data.plot.bar(y='Recall', figsize=(14,6))
    ax.set_xticklabels(data.iloc[:, 0])
    plt.tight_layout()
    plt.savefig(f"figures/RperClass_{version}.pdf")
    plt.show()

def plotF1(data, version):
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
    plt.savefig(f"figures/F1_{version}.pdf")
    print(f"Optimal K: {optimal+1}")
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', help='Which data to visualize ("prcurve" or "prperclass" or "f1") (default=prcurve)', default='prcurve')
    parser.add_argument('--version', help='Which version of the query algorithm to visualize ("custom" or "ANN") (default=custom)', default='custom')
    args = parser.parse_args()

    if args.type == 'prcurve':
        data = pd.read_csv(f"pr_cache_{args.version}.csv")
        plotPRCurve(data, args.version)
    elif args.type == 'f1':
        data = pd.read_csv(f"pr_cache_{args.version}.csv")
        plotF1(data, args.version)
    elif args.type == 'prperclass':
        data = pd.read_csv(f"pr_per_class_{args.version}.csv")
        plotPRperClass(data, args.version)