# for 2 files at once
# histograms or spectograms for a selected frequency
# for all the files at once or all the files in one plot
#mean meadian mode of each plot
import os
import numpy as np
import mmap
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import mode
from scipy.signal import spectrogram

# directory_path = "/media/oshani/Shared/UBUNTU/EMforTomography/893/no_object"


# Directory path containing .cfile files
file_path1 = "/media/oshani/Shared/UBUNTU/EMforTomography/794/samindu"
file_path2 = "/media/oshani/Shared/UBUNTU/EMforTomography/794/no_object"

# Parameters  Hz
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


# Plot the signal strength of targted frequency over time
def plot_spectrogram_for_targeted(target_bin_values, k, label):
    plt.plot(target_bin_values, linewidth=0.2, alpha=0.9, label=label)    
    plt.xlabel("Time [s]")
    plt.ylabel("Signal Strength [dB]")
    # plt.title(f"Signal Strength at {target_freq / 1e6} MHz from {k*2} feet distance")
    plt.title(f"Signal Strength at {target_freq / 1e6} MHz ")
    plt.grid(True)
    plt.legend()


def plot_histogram_for_targeted(target_bin_values, label, title, alpha_mean, alpha_median, alpha_mode):
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
    plt.axvline(mean_value, color='red', alpha=alpha_mean, linestyle='dashed', linewidth=2, label=f"Mean: {mean_value:.2f} dB")
    plt.axvline(median_value, color='green', alpha=alpha_median, linestyle='dashed', linewidth=2, label=f"Median: {median_value:.2f} dB")
    plt.axvline(mode_value, color='blue', alpha=alpha_mode, linestyle='dashed', linewidth=2, label=f"Mode: {mode_value:.2f} dB")

    plt.legend()

    

def stat_for_targeted(target_bin_values):
    #see more
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


def all_stat(cfile_files):

    print(cfile_files)
    mean_values = []
    median_values = []
    mode_values = []

    for i, cfile in enumerate(cfile_files, start=1):
        file_path = os.path.join(directory_path, cfile)
        
        time, binvalue = process_iq_data(file_path)

        mean_value, median_value, mode_value = stat_for_targeted(binvalue)

        mean_values.append(mean_value)
        median_values.append(median_value)
        mode_values.append(mode_value)

    # Now, create a separate plot showing the variation of mean, median, and mode with respect to k (i.e., the file index)
    plt.figure(figsize=(12, 6))

    # Plot the mean, median, and mode as a function of k (file index)
    plt.plot(range(1, len(cfile_files) + 1), mean_values, label='Mean', marker='o', color='red', linestyle='-', markersize=5)
    plt.plot(range(1, len(cfile_files) + 1), median_values, label='Median', marker='o', color='green', linestyle='-', markersize=5)
    plt.plot(range(1, len(cfile_files) + 1), mode_values, label='Mode', marker='o', color='blue', linestyle='-', markersize=5)

    # Add labels and title
    plt.xlabel('File Index (k)', fontsize=12)
    xticks = plt.gca().get_xticks()
    plt.xticks(xticks, [2 * int(x) for x in xticks])
    plt.ylabel('Power (dB)', fontsize=12)
    plt.title(f'Variation of Mean, Median, and Mode for {target_freq / 1e6} MHz  with Distance in feet', fontsize=14)
    plt.legend()
    plt.grid(True)

    # Save the variation plot
    variation_output_path = os.path.join(directory_path, "allstat2.png")
    # plt.savefig(variation_output_path)
    plt.show()


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


#modifed to claculate mmm values of each segment
def process_iq_data_over1file(file_path, interval_duration=1):
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


def all_stat_for_every_second(file_path, cfile_files, label,color):

    # cfile_files=['794_8t_null.cfile']
    
    for i, cfile in enumerate(cfile_files, start=1):
        file_path = os.path.join(file_path, cfile)
        print(f"Processing file: {cfile}")

        
        mean_values , median_values, mode_values = process_iq_data_over1file(file_path)

        # Plot Mean values for the file
        # plt.figure(figsize=(10, 5))
        # plt.plot(range(1, len(mean_values) + 1), mean_values, label='Mean', marker='o', color='red', linestyle='-')
        plt.plot(range(1, len(median_values) + 1), median_values, label=label, marker='o', color=color, linestyle='-')
        # plt.plot(range(1, len(mode_values) + 1), mode_values, label='Mode', marker='o', color='blue', linestyle='-')

        plt.xlabel('Time Interval (Seconds)', fontsize=12)
        plt.ylabel('Power (dB)', fontsize=12)
        plt.title(f'Power Variation for {cfile}', fontsize=14)
        plt.grid(True)
        # plt.savefig(os.path.join(directory_path, f"mmmeverysecond_{cfile}.png"))
        plt.legend()
        # plt.show()

# # for spectrogram and histogram calculation
# time1,target_bin1=process_iq_data(file_path1)
# time2,target_bin2=process_iq_data(file_path2)

# plt.figure(figsize=(20, 10))

# # plot_histogram_for_targeted(target_bin1, label="with object", title=f"Histogram of Power Values at {target_freq / 1e6} MHz", alpha_mean=0.7, alpha_median=0.7, alpha_mode=0.7)
# # plot_histogram_for_targeted(target_bin2, label="with out object", title=f"Histogram of Power Values at {target_freq / 1e6} MHz", alpha_mean=0.7, alpha_median=0.7, alpha_mode=0.7)

# # plot_spectrogram_for_targeted(target_bin1, k=1, label="with ")
# # plot_spectrogram_for_targeted(target_bin2, k=1, label="with object")


# plt.show()


# # for mmm calculation
# cfile_files = [f for f in os.listdir(directory_path) if f.endswith('.cfile')]
# print(cfile_files)

# sorted_cfile_files = sorted(cfile_files, key=lambda x: int(x.split('_')[1].split('t')[0]))

# print(sorted_cfile_files)

# all_stat(sorted_cfile_files)


plt.figure(figsize=(10, 5))

cfile_files=['794_10t_null.cfile']
all_stat_for_every_second(file_path2,cfile_files,label="median without object",color="green")

cfile_files=['794_10t_.cfile']
all_stat_for_every_second(file_path1,cfile_files,label="median with object",color="blue")

plt.show()