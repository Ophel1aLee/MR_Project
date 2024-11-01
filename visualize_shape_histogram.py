import pandas as pd
import matplotlib.pyplot as plt


def plot_histograms_from_csv(csv_file, model_filename):
    df = pd.read_csv(csv_file)

    model_data = df[df['file_name'] == model_filename]
    if model_data.empty:
        print(f"Model name '{model_filename}' doesn't exist in the CSV file.")
        return

    histograms = []
    histograms.append(model_data.iloc[0, 8:108].values)
    for i in range(1, 5):  # 另外四个直方图
        start_idx = 108 + (i - 1) * 100
        end_idx = start_idx + 100
        histograms.append(model_data.iloc[0, start_idx:end_idx].values)

    for i, histogram in enumerate(histograms):
        plt.figure()
        plt.bar(range(100), histogram)
        plt.title(['A3', 'D1', 'D2', 'D3', 'D4'][i])
        plt.tight_layout()
        plt.show()