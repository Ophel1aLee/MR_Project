import pandas as pd


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


histogram_normalizing("descriptors.csv", 100)