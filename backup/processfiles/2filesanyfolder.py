import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import mmap

# File paths
iq_data_file1 = "/media/oshani/Shared/UBUNTU/EMforTomography/794/dasun/794_3t.cfile"
iq_data_file2 = "/media/oshani/Shared/UBUNTU/EMforTomography/893/no_object/893_3t_null.cfile"

# Parameters
sampling_frequency = 20e6  # Hz
center_frequency = 893e6  # Hz

target_freq = 890e6

start_time1 = 0
end_time1 = 3
start_time2 = 0
end_time2 = 3

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

def compute_spectrogram(iq_segment):
    """
    Computes the spectrogram and filters the target frequency.
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

    target_index = np.abs(frequencies_shifted - target_freq).argmin()
    print("Spectrogram computation complete.")

    # Checking the type and size of the returned structures
    print(f"Type of times: {type(times)}")
    print(f"Size of times: {times.shape if isinstance(times, np.ndarray) else len(times)}")

    print(f"Type of Sxx: {type(Sxx)}")
    print(f"Size of Sxx: {Sxx.shape if isinstance(Sxx, np.ndarray) else len(Sxx)}")


    return times, 10 * np.log10(Sxx[target_index, :])  # Convert power to dB

def process_iq_data(file_path, interval_duration=1):
    """
    Processes IQ data from a file, extracting and plotting spectrogram segments.
    """
    iq_data = read_iq_data(file_path)
    total_samples = len(iq_data)
    samples_per_interval = int(sampling_frequency * interval_duration)

    interval_index = 0
    temp_target_bin_segments=np.array([])
    temp_time=np.array([])

    while interval_index * samples_per_interval < total_samples:
        start_sample = interval_index * samples_per_interval
        end_sample = start_sample + samples_per_interval

        # Extract segment and plot spectrogram
        iq_segment = iq_data[start_sample:end_sample]
        target_time, target_bin_segments=compute_spectrogram(iq_segment)

        # Concatenate the computed segments to the temporary arrays
        temp_target_bin_segments = np.concatenate((temp_target_bin_segments, target_bin_segments))
        temp_time = np.concatenate((temp_time, target_time))


        interval_index += 1

    return temp_time,temp_target_bin_segments



def stat_for_targeted(target_bin_values):
    no_of_bins  = int(np.sqrt(len(target_bin_values)))
    print(no_of_bins)

    counts, bin_edges = np.histogram(target_bin_values, bins=no_of_bins)

    # Calculate mean from histogram bin centers and counts (weighted mean)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2  # Calculate the bin centers
    mean_value = np.sum(bin_centers * counts) / np.sum(counts)  # Weighted mean

    # Calculate median from cumulative distribution
    cumulative_counts = np.cumsum(counts)  # Cumulative sum of counts
    median_bin_index = np.searchsorted(cumulative_counts, cumulative_counts[-1] / 2)  # Find the bin with cumulative count > 50%
    median_value = bin_centers[median_bin_index]

    # Calculate mode as the bin center with the highest count
    mode_value = bin_centers[np.argmax(counts)]

    return mean_value, median_value, mode_value

def plot_spectrogram(time, target_bin_values, label, color):

    time = np.arange(len(target_bin_values)) / (2*sampling_frequency)

   
    plt.plot(time, target_bin_values, linewidth=0.5, label=label, color=color)    
    plt.xlabel("Time [s]")
    plt.ylabel("Signal Strength [dB]")
    plt.title(f"Signal Strength at {target_freq / 1e6} MHz")
    plt.grid(True)
    plt.legend()


#provide title whether this mark mean,median, mode or none
def plot_histogram_for_targeted(target_bin_values, title, alpha_mean, alpha_median, alpha_mode):
    no_of_bins = int(np.sqrt(len(target_bin_values)))
    print(no_of_bins)

    plt.hist(target_bin_values, bins=no_of_bins, alpha=0.7)
    plt.xlabel("Power (dB)")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.grid(True)

    mean_value, median_value, mode_value = stat_for_targeted(target_bin_values)

    # Mark the mean, median, and mode on the histogram with specified alpha values
    plt.axvline(mean_value, color='red', alpha=alpha_mean, linestyle='dashed', linewidth=2, label=f"Mean: {mean_value:.2f} dB")
    plt.axvline(median_value, color='green', alpha=alpha_median, linestyle='dashed', linewidth=2, label=f"Median: {median_value:.2f} dB")
    plt.axvline(mode_value, color='blue', alpha=alpha_mode, linestyle='dashed', linewidth=2, label=f"Mode: {mode_value:.2f} dB")

    plt.legend()



time1,target_bin1=process_iq_data(iq_data_file1)
time2,target_bin2=process_iq_data(iq_data_file2)

plt.figure(figsize=(20, 10))

# plot_spectrogram(time1,target_bin1, label="with", color="blue")
# plot_spectrogram(time2,target_bin2, label="without", color="orange")

plot_histogram_for_targeted(target_bin1, title="test", alpha_mean=0.7, alpha_median=0.7, alpha_mode=0.7)
plot_histogram_for_targeted(target_bin2, title="test", alpha_mean=0.7, alpha_median=0.7, alpha_mode=0.7)



plt.show()