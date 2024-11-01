import pandas as pd
import matplotlib.pyplot as plt
from streamlit import columns

from standardization import columns_to_standardize, means


def histogram_normalizing(csv_file, output_csv_file, histogram_bins):
    df = pd.read_csv(csv_file)
    num_bins = histogram_bins
    num_histograms = 5

    for i in range(num_histograms):
        histogram_cols = df.iloc[:, 8 + i * num_bins : 8 + (i + 1) * num_bins]
        sum_hist = histogram_cols.sum(axis=1)
        df.iloc[:, 8 + i * num_bins: 8 + (i + 1) * num_bins] = histogram_cols.div(sum_hist, axis=0)

    df.to_csv(output_csv_file, index=False)
    print("Histogram normalizing finished.")


def plot_histograms_for_class(csv_file, class_name, histogram_bins):
    df = pd.read_csv(csv_file)
    # Filter the DataFrame for the given class name
    class_column_name = df.columns[0]
    class_data = df[df[class_column_name] == class_name]
    num_histograms = 5
    histogram_names = ['A3', 'D1', 'D2', 'D3', 'D4']

    if class_data.empty:
        print(f"Class '{class_name}' not found in the dataset.")
        return

    # Plot each histogram type for all models of the given class
    plt.figure(figsize=(15, 10))
    for i in range(num_histograms):
        plt.subplot(num_histograms, 1, i + 1)
        for index, row in class_data.iterrows():
            histogram_cols = row[8 + i * histogram_bins: 8 + (i + 1) * histogram_bins]
            plt.plot(histogram_cols, alpha=0.7)
        plt.title(f'Histogram {histogram_names[i]} for class: {class_name}')
        plt.xlabel('Bins')
        plt.ylabel('Normalized Frequency')
    plt.tight_layout()
    plt.show()


def single_value_normalizing(csv_file, output_csv_file):
    df = pd.read_csv(csv_file)
    columns_to_standardize = df.columns[2:8]
    for column in columns_to_standardize:
        mean = df[column].mean()
        std = df[column].std()
        df[column] = (df[column] - mean) /std

    df.to_csv(output_csv_file, index=False)
    print("Single value normalizing finished.")