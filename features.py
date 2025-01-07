import numpy as np
from scipy.fft import fft
from scipy.signal import welch

def mean(window):
    ax = window['ax'].mean()
    ay = window['ay'].mean()
    az = window['az'].mean()
    return ax, ay, az

def root_mean_square(window):
    rms_ax = np.sqrt(np.mean(window['ax']**2))
    rms_ay = np.sqrt(np.mean(window['ay']**2))
    rms_az = np.sqrt(np.mean(window['az']**2))
    return rms_ax, rms_ay, rms_az

def standard_deviation(window):
    std_ax = np.std(window['ax'])
    std_ay = np.std(window['ay'])
    std_az = np.std(window['az'])
    return std_ax, std_ay, std_az

def pearson_correlation(window):
    corr_xy = window['ax'].corr(window['ay'])
    corr_xz = window['ax'].corr(window['az'])
    corr_yx = window['ay'].corr(window['ax'])
    return corr_xy, corr_xz, corr_yx

def signal_magnitude_vector(window):
    return np.sqrt(window['ax']**2 + window['ay']**2 + window['az']**2)

def interquartile_range(window):
    iqr_ax = window['ax'].quantile(0.75) - window['ax'].quantile(0.25)
    iqr_ay = window['ay'].quantile(0.75) - window['ay'].quantile(0.25)
    iqr_az = window['az'].quantile(0.75) - window['az'].quantile(0.25)
    return iqr_ax, iqr_ay, iqr_az

def max_min(window):
    max_ax = window['ax'].max() - window['ax'].min()
    max_ay = window['ay'].max() - window['ay'].min()
    max_az = window['az'].max() - window['az'].min()
    return max_ax, max_ay, max_az

def kurtosis(window):
    kurtosis_ax = window['ax'].kurtosis()
    kurtosis_ay = window['ay'].kurtosis()
    kurtosis_az = window['az'].kurtosis()
    return kurtosis_ax, kurtosis_ay, kurtosis_az

def skewness(window):
    skewness_ax = window['ax'].skew()
    skewness_ay = window['ay'].skew()
    skewness_az = window['az'].skew()
    return skewness_ax, skewness_ay, skewness_az

def mean_frequency_power(window):
    freq_ax = np.mean(np.abs(fft(window['ax'])))
    freq_ay = np.mean(np.abs(fft(window['ay'])))
    freq_az = np.mean(np.abs(fft(window['az'])))
    return freq_ax, freq_ay, freq_az


#------------------------------------------PSD------------------------------------------
def compute_psd_label(signal, fs):
    freqs, psd = welch(signal, fs=fs, nperseg=len(signal))
    dominant_freq = freqs[np.argmax(psd)]
    return dominant_freq, 1 if 4 <= dominant_freq <= 8 else 0