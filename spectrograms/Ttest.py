import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from scipy import stats
import mmap
from scipy.stats import mode

# File paths
iq_data_file1 = "/media/oshani/Shared/UBUNTU/EMforTomography/data-sandali/12tile/12tilewsandali893.cfile"
start_time1 = 3
end_time1 = 4

iq_data_file2 = "/media/oshani/Shared/UBUNTU/EMforTomography/data-sandali/12tile/12tilewosandali893.cfile"
start_time2 = 4
end_time2 = 5

# Parameters
sampling_frequency = 20e6  # Hz
center_frequency = 893e6  # Hz
# start_time = 2  # seconds
# end_time = 3    # seconds
freq_min = 890e6  # Hz
freq_max = 892e6  # Hz
target_frequency = 891e6  # Hz
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
            raw_data = np.frombuffer(mm, dtype=np.float32).copy()
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
    
    return times, frequencies_shifted[mask], 10 * np.log10(Sxx[mask, :] + 1e-12)  # Convert power to dB

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
    print(f"  Minimum Power (dB): {min_power:.4f}")
    print(f"  Maximum Power (dB): {max_power:.4f}")

def perform_t_test(sample_A, sample_B, alpha):
    """
    Performs an independent sample t-test and prints results.
    """
    t_statistic, p_value = stats.ttest_ind(sample_A, sample_B)
    df = len(sample_A) + len(sample_B) - 2
    critical_t = stats.t.ppf(1 - alpha / 2, df)

    print(f"T-Statistic: {t_statistic:.4f}")
    print(f"P-Value: {p_value:.4e}")
    print(f"Critical T-Value: {critical_t:.4f}")

    if np.abs(t_statistic) > critical_t:
        print("T-Test: Significant difference between the groups.")
    else:
        print("T-Test: No significant difference found between the groups.")

    if p_value < alpha:
        print("P-Value: Evidence found to reject the null hypothesis.")
    else:
        print("P-Value: No evidence to reject the null hypothesis.")


def visualize_data(sample_A, sample_B, times, label_A="Sample A", label_B="Sample B"):
    """
    Visualizes the power values for two samples using line plots, histograms, and box plots.
    """
    # Line plot of power values over time
    plt.figure(figsize=(12, 6))
    plt.plot(times, sample_A, label=label_A, color='blue', alpha=0.7)
    plt.plot(times, sample_B, label=label_B, color='orange', alpha=0.7)
    plt.xlabel("Time (s)")
    plt.ylabel("Power (dB)")
    plt.title("Power Over Time")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Histogram of power values
    plt.figure(figsize=(12, 6))
    plt.hist(sample_A, bins=30, alpha=0.7, label=label_A, color='blue')
    plt.hist(sample_B, bins=30, alpha=0.7, label=label_B, color='orange')
    plt.xlabel("Power (dB)")
    plt.ylabel("Frequency")
    plt.title("Histogram of Power Values")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Box plot of power values
    plt.figure(figsize=(8, 6))
    plt.boxplot([sample_A, sample_B], tick_labels=[label_A, label_B], patch_artist=True, 
                boxprops=dict(facecolor='lightblue'), medianprops=dict(color='red'))
    plt.ylabel("Power (dB)")
    plt.title("Box Plot of Power Values")
    plt.grid(True)
    plt.show()


def visualize_data_combined(sample_A, sample_B, times, label_A="Sample A", label_B="Sample B"):
    """
    Visualizes the power values for two samples using subplots for line plot, histogram, and box plot.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 18))
    fig.tight_layout(pad=6.0)

    # Line plot of power values over time
    axes[0].plot(times, sample_A, label=label_A, color='blue', alpha=0.7)
    axes[0].plot(times, sample_B, label=label_B, color='orange', alpha=0.7)
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Power (dB)")
    axes[0].set_title("Power Over Time")
    axes[0].legend()
    axes[0].grid(True)

    # Histogram of power values
    axes[1].hist(sample_A, bins=1000, alpha=0.7, label=label_A, color='blue')
    axes[1].hist(sample_B, bins=1000, alpha=0.7, label=label_B, color='orange')
    axes[1].set_xlabel("Power (dB)")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Histogram of Power Values")
    axes[1].legend()
    axes[1].grid(True)

    # Box plot of power values
    axes[2].boxplot([sample_A, sample_B], tick_labels=[label_A, label_B], patch_artist=True, 
                    boxprops=dict(facecolor='lightblue'), medianprops=dict(color='red'))
    axes[2].set_ylabel("Power (dB)")
    axes[2].set_title("Box Plot of Power Values")
    axes[2].grid(True)

    plt.show()

# Main Execution
print("Starting analysis...")

# Read IQ data
iq_data1 = read_iq_data(iq_data_file1)
iq_data2 = read_iq_data(iq_data_file2)

# Extract segments
segment1 = extract_segment(iq_data1, start_time1, end_time1, sampling_frequency)
segment2 = extract_segment(iq_data2, start_time2, end_time2, sampling_frequency)

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

# Visualize the data
# visualize_data(sample_A, sample_B, times1, label_A="Sample A (File 1)", label_B="Sample B (File 2)")

# Visualize the data
visualize_data_combined(sample_A, sample_B, times1, label_A="Sample A (File 1)", label_B="Sample B (File 2)")

print("Analysis complete.")
