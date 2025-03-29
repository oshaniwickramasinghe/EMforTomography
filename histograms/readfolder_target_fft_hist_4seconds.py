#process files for each second
#create mmm plot for each second of a file
#create histogram for each second of a file
import os
import numpy as np
import mmap
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import mode
from scipy.signal import spectrogram


# # Directory path containing .cfile files
# directory_path = "/media/oshani/Shared/UBUNTU/EMforTomography/dec_20_rooftop/893/samindu"

# # Parameters  Hz
# sampling_frequency = 20e6  
# center_frequency = 893e6  
# target_freq = 891e6


# directory_path = "/media/oshani/Shared/UBUNTU/EMforTomography/dec_20_rooftop/827/samindu"
# sampling_frequency = 20e6  
# center_frequency = 827e6  
# target_freq = 825e6

directory_path = "/media/oshani/Shared/UBUNTU/EMforTomography/sensorlogs/compare"
sampling_frequency = 20e6  
center_frequency = 794e6  
target_freq = 792e6


def read_iq_data(file_path):
    print(f"Reading IQ data from: {file_path}")
    with open(file_path, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            file_size = mm.size()
            num_samples = file_size // (2 * np.dtype(np.float32).itemsize)
            raw_data = np.frombuffer(mm, dtype=np.float32).copy()
            iq_data = raw_data[0::2] + 1j * raw_data[1::2]
    print("Finished reading IQ data.")
    return iq_data


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

    return times, 10 * np.log10(Sxx[target_index, :])  # Convert power to dB
    # return times, 10 * np.log10(Sxx[target_index, :] + 1e-12)  # Convert power to dB

  

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

def plot_histogram_for_targeted(target_bin_values, k, label, title, alpha_mean, alpha_median, alpha_mode):
    no_of_bins = int(np.sqrt(len(target_bin_values)))
    print(no_of_bins)

    # Plot the histogram with the specified alpha value
    plt.hist(target_bin_values, bins=no_of_bins, alpha=0.7, label=label)
    plt.xlabel("Power (dB)")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.grid(True)

    mean_value, median_value, mode_value = stat_for_targeted(target_bin_values)

    # Mark the mean, median, and mode on the histogram with specified alpha values
    # plt.axvline(mean_value, color='red', alpha=alpha_mean, linestyle='dashed', linewidth=2, label=f"Mean: {mean_value:.2f} dB")
    plt.axvline(median_value, color='green', alpha=alpha_median, linestyle='dashed', linewidth=2, label=f"Median: {median_value:.2f} dB")
    # plt.axvline(mode_value, color='blue', alpha=alpha_mode, linestyle='dashed', linewidth=2, label=f"Mode: {mode_value:.2f} dB")

    plt.legend()


#modifed to claculate mmm values of each segment
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

    mean_values = []
    median_values = []
    mode_values = []

    while interval_index * samples_per_interval < total_samples:
        start_sample = interval_index * samples_per_interval
        end_sample = start_sample + samples_per_interval

        # Extract segment and plot spectrogram
        iq_segment = iq_data[start_sample:end_sample]
        target_time, target_bin_segments=compute_spectrogram(iq_segment)

        mean_value, median_value, mode_value = stat_for_targeted(target_bin_segments)

        mean_values.append(mean_value)
        median_values.append(median_value)
        mode_values.append(mode_value)
        
        # Concatenate the computed segments to the temporary arrays
        temp_target_bin_segments = np.concatenate((temp_target_bin_segments, target_bin_segments))
        temp_time = np.concatenate((temp_time, target_time))


        interval_index += 1

    # return temp_time,temp_target_bin_segments
    return mean_values, median_values, mode_values


# Function to process IQ data and plot statistics for every second of each file
def all_stat_for_every_second(cfile_files):

    cfile_files=['60s_1f_794_idle.cfile']
    
    for i, cfile in enumerate(cfile_files, start=1):
        file_path = os.path.join(directory_path, cfile)
        print(f"Processing file: {cfile}")

        
        mean_values , median_values, mode_values = process_iq_data(file_path)

        # Plot Mean values for the file
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(mean_values) + 1), mean_values, label='Mean', marker='o', color='red', linestyle='-')
        plt.plot(range(1, len(median_values) + 1), median_values, label='Median', marker='o', color='green', linestyle='-')
        plt.plot(range(1, len(mode_values) + 1), mode_values, label='Mode', marker='o', color='blue', linestyle='-')

        plt.xlabel('Time Interval (Seconds)', fontsize=12)
        plt.ylabel('Power (dB)', fontsize=12)
        plt.title(f'Power Variation for {cfile}', fontsize=14)
        plt.grid(True)
        plt.legend()
        download_directory_path = "/media/oshani/Shared/UBUNTU/EMforTomography/images/mmmeverysecond/"
        plt.savefig(os.path.join(download_directory_path, f"mmmeverysecond_{cfile}.png"))
        
        # plt.show()



cfile_files = [f for f in os.listdir(directory_path) if f.endswith('.cfile')]
print(cfile_files)

# sorted_cfile_files = sorted(cfile_files, key=lambda x: int(x.split('_')[1].split('t')[0]))

# print(sorted_cfile_files)

all_stat_for_every_second(cfile_files)


