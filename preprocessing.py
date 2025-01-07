import pandas as pd
from features import *


data1 = pd.read_excel('Data/1.xlsx')
data2 = pd.read_excel('Data/2.xlsx')
data3 = pd.read_excel('Data/3.xlsx')
data4 = pd.read_excel('Data/4.xlsx')
data5 = pd.read_excel('Data/5.xlsx')

data = [data1,data2,data3,data4,data5]


def moving_window_feature_extraction(data, window_size, step_size=1, fs=50):

    features_list = []
    total_samples = len(data)

    # Itération sur les fenêtres
    for start in range(0, total_samples - window_size + 1, step_size):
        end = start + window_size
        window = data.iloc[start:end]

        mean_values = mean(window)
        rms_values = root_mean_square(window)
        std_values = standard_deviation(window)
        corr_values = pearson_correlation(window)
        iqr_values = interquartile_range(window)
        max_min_values = max_min(window)
        kurtosis_values = kurtosis(window)
        skewness_values = skewness(window)
        mean_freq_power_values = mean_frequency_power(window)

        smv_values = signal_magnitude_vector(window)
        smv_mean = smv_values.mean()
        smv_max = smv_values.max()
        smv_min = smv_values.min()

        dominant_freq_ax, label_ax = compute_psd_label(window['ax'], fs)
        dominant_freq_ay, label_ay = compute_psd_label(window['ay'], fs)
        dominant_freq_az, label_az = compute_psd_label(window['az'], fs)

        label_global = 1 if (label_ax + label_ay + label_az) >= 2 else 0


        features = {
            'start_index': start,
            'end_index': end,
            'mean_ax': mean_values[0],
            'mean_ay': mean_values[1],
            'mean_az': mean_values[2],
            'rms_ax': rms_values[0],
            'rms_ay': rms_values[1],
            'rms_az': rms_values[2],
            'std_ax': std_values[0],
            'std_ay': std_values[1],
            'std_az': std_values[2],
            'corr_xy': corr_values[0],
            'corr_xz': corr_values[1],
            'corr_yz': corr_values[2],
            'smv_mean': smv_mean,
            'smv_max': smv_max,
            'smv_min': smv_min,
            'iqr_ax': iqr_values[0],
            'iqr_ay': iqr_values[1],
            'iqr_az': iqr_values[2],
            'max_min_ax': max_min_values[0],
            'max_min_ay': max_min_values[1],
            'max_min_az': max_min_values[2],
            'kurtosis_ax': kurtosis_values[0],
            'kurtosis_ay': kurtosis_values[1],
            'kurtosis_az': kurtosis_values[2],
            'skewness_ax': skewness_values[0],
            'skewness_ay': skewness_values[1],
            'skewness_az': skewness_values[2],
            'mean_freq_power_ax': mean_freq_power_values[0],
            'mean_freq_power_ay': mean_freq_power_values[1],
            'mean_freq_power_az': mean_freq_power_values[2],
            'dominant_freq_ax': dominant_freq_ax,
            'dominant_freq_ay': dominant_freq_ay,
            'dominant_freq_az': dominant_freq_az,
            'label_ax': label_ax,
            'label_ay': label_ay,
            'label_az': label_az,
            'label': label_global
        }

        features_list.append(features)

    # Conversion en DataFrame
    features_df = pd.DataFrame(features_list)

    return features_df

def preprocess_data(data, window_size, step_size, fs):
    for i in range(len(data)):
        features_df = moving_window_feature_extraction(data[i], window_size, step_size, fs)
        # save data to csv named data_i.csv in Data_Preprocessed folder
        features_df.to_csv('Data_Preprocessed/data_'+str(i)+'.csv', index=False)



if __name__ == '__main__':
    window_size = 128
    step_size = 64
    fs = 50
    preprocess_data(data, window_size, step_size, fs)