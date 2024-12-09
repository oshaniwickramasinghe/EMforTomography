#this works 
# #change 
# center_frequency = 299e6   # MHz
# start_time = 5  # seconds
# end_time = 7   # seconds
# freq_min = 296e6
# freq_max = 298e6
# title="sampling range = 20MHz,  center frequency = 299MHz,  time= from 5s to 7s",
# target frequency

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from scipy import stats
import mmap
from scipy.stats import mode


# File paths
iq_data_file1 = "/media/oshani/Shared/UBUNTU/EMforTomography/data-sandali/sandali moves from 3/12tilewosandali794.cfile"
iq_data_file2 = "/media/oshani/Shared/UBUNTU/EMforTomography/data-sandali/sandali moves from 3/Aat12tilewsandaliat10794.cfile"

# Parameters
sampling_frequency = 20e6  # do not change
center_frequency = 794e6   #  MHz
start_time = 2  # seconds
end_time = 3    # seconds
freq_min = 791e6
freq_max = 793e6  # Frequency range of interest
target_frequency = 792e6  # Frequency of interest
alpha = 0.05  # Significance level for hypothesis testing

def read_iq_data(file_path):
    """
    Reads IQ data from a binary file and converts it to complex values.
    """
    print(f"Reading IQ data from: {file_path}")
    with open(file_path, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            file_size = mm.size()
            num_samples = file_size // (2 * np.dtype(np.float32).itemsize)
            raw_data = np.frombuffer(mm, dtype=np.float32).copy()  # Copy to avoid BufferError
            iq_data = raw_data[0::2] + 1j * raw_data[1::2]
    print("Finished reading IQ data.")
    return iq_data

def extract_segment(iq_data, start_time, end_time, sampling_frequency):
    """
    Extracts a specific time segment from IQ data.
    """
    start_sample = int(start_time * sampling_frequency)
    end_sample = int(end_time * sampling_frequency)
    return iq_data[start_sample:end_sample]

def compute_spectrogram(iq_segment, sampling_frequency, center_frequency, freq_min, freq_max):
    """
    Computes the spectrogram and filters frequencies within a specific range.
    """
    print("Computing spectrogram...")
    frequencies, times, Sxx = spectrogram(
        iq_segment,
        fs=sampling_frequency,
        window='hann',
        nperseg=1024,
        noverlap=512,
        nfft=2048,
        scaling='density'
    )
    frequencies_shifted = frequencies + (center_frequency - sampling_frequency / 2)
    mask = (frequencies_shifted >= freq_min) & (frequencies_shifted <= freq_max)
    print("Spectrogram computation complete.")
    
    return times, frequencies_shifted[mask], 10 * np.log10(Sxx[mask, :] + 1e-12)  # Convert power to dB scale

def print_statistical_values(sample, label="Sample"):
    """
    Computes and prints statistical values for the given sample.
    """
    mean_value = np.mean(sample)
    median_value = np.median(sample)
    std_dev = np.std(sample)
    variance = np.var(sample)
    mode_value = mode(sample, keepdims=True)
    min_power = np.min(sample)
    max_power = np.max(sample)

    print(f"Statistical values for {label}:")
    print(f"  Mean: {mean_value:.4f} dB")
    print(f"  Median: {median_value:.4f} dB")
    print(f"  Standard Deviation: {std_dev:.4f} dB")
    print(f"  Variance: {variance:.4f} dB^2")
    print(f"  Mode: {mode_value.mode[0]:.4f} dB (Frequency: {mode_value.count[0]} occurrences)")
    print(f"  Minimum Power (dB): {min_power}")
    print(f"  Maximum Power (dB): {max_power}")



def perform_t_test(sample_A, sample_B, alpha):
    """
    Performs an independent sample t-test and prints results.
    """
    t_statistic, p_value = stats.ttest_ind(sample_A, sample_B)
    df = len(sample_A) + len(sample_B) - 2
    critical_t = stats.t.ppf(1 - alpha / 2, df)

    print(f"T-Statistic: {t_statistic}")
    print(f"P-Value: {p_value}")
    print(f"Critical T-Value: {critical_t}")
    if np.abs(t_statistic) > critical_t:
        print("With T-value: Significant difference between the groups.")
    else:
        print("With T-value: No significant difference found between the groups.")

    if p_value < alpha:
        print("With P-value: Evidence found to reject the null hypothesis.")
    else:
        print("With P-value: No evidence to reject the null hypothesis.")


# Read IQ data
iq_data1 = read_iq_data(iq_data_file1)
iq_data2 = read_iq_data(iq_data_file2)

# Extract segments
segment1 = extract_segment(iq_data1, start_time, end_time, sampling_frequency)
segment2 = extract_segment(iq_data2, start_time, end_time, sampling_frequency)

# Compute spectrograms and extract power values for the target frequency
times1, frequencies1, Sxx1_dB = compute_spectrogram(segment1, sampling_frequency, center_frequency, freq_min, freq_max)
times2, frequencies2, Sxx2_dB = compute_spectrogram(segment2, sampling_frequency, center_frequency, freq_min, freq_max)

# Find the closest frequency index
freq_index1 = np.abs(frequencies1 - target_frequency).argmin()
freq_index2 = np.abs(frequencies2 - target_frequency).argmin()

# Extract the power values for the selected frequency bin
sample_A = Sxx1_dB[freq_index1, :]
sample_B = Sxx2_dB[freq_index2, :]

# Print statistical values for sample_A and sample_B
print_statistical_values(sample_A, label="Sample A (File 1)")
print_statistical_values(sample_B, label="Sample B (File 2)")

# Perform T-Test
perform_t_test(sample_A, sample_B, alpha)
