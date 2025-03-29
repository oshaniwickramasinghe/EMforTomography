# read files in 3 diferent folders, calculate mmm for eah file 
# plot one of the m for each file, 
# to see how mmm changes with time with or without an object
import os
import numpy as np
import mmap
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import mode
from scipy.signal import spectrogram


# # Directory path containing .cfile files
# directory_path1 = "/media/oshani/Shared/UBUNTU/EMforTomography/893/dasun"
# directory_path2 = "/media/oshani/Shared/UBUNTU/EMforTomography/893/samindu"
# directory_path3 = "/media/oshani/Shared/UBUNTU/EMforTomography/893/no_object"
# Parameters  Hz
# sampling_frequency = 20e6  
# center_frequency = 893e6  
# target_freq = 891e6


# directory_path1 = "/media/oshani/Shared/UBUNTU/EMforTomography/827/dasun"
# directory_path2 = "/media/oshani/Shared/UBUNTU/EMforTomography/827/samindu"
# directory_path3 = "/media/oshani/Shared/UBUNTU/EMforTomography/827/no_object"
# sampling_frequency = 20e6  
# center_frequency = 827e6  
# target_freq = 825e6

# directory_path1 = "/media/oshani/Shared/UBUNTU/EMforTomography/794/dasun"
# directory_path2 = "/media/oshani/Shared/UBUNTU/EMforTomography/794/samindu"
# directory_path3 = "/media/oshani/Shared/UBUNTU/EMforTomography/794/no_object"

directory_path1 = "/media/oshani/Shared/UBUNTU/EMforTomography/waru/794/withoutwaru/use"
# directory_path2 = "/media/oshani/Shared/UBUNTU/EMforTomography/waru/794/withoutwaru"
# directory_path3 = "/media/oshani/Shared/UBUNTU/EMforTomography/ground/off"

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
    plt.title(f"Signal Strength at {target_freq / 1e6} MHz from {k*2} feet distance")
    # plt.title(f"Signal Strength at {target_freq / 1e6} MHz ")
    plt.grid(True)
    plt.legend()

    

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


def all_stat(sorted_cfile_files1):

    mean_values1 = []
    median_values1 = []
    mode_values1 = []

    mean_values2 = []
    median_values2 = []
    mode_values2 = []

    mean_values3 = []
    median_values3 = []
    mode_values3 = []

    # Now, create a separate plot showing the variation of mean, median, and mode with respect to k (i.e., the file index)
    plt.figure(figsize=(12, 6))

    for i, cfile in enumerate(sorted_cfile_files1, start=1):
        file_path = os.path.join(directory_path1, cfile)
        
        time, target_bin_values = process_iq_data(file_path)

        mean_value, median_value, mode_value = stat_for_targeted(target_bin_values)

        mean_values1.append(mean_value)
        median_values1.append(median_value)
        mode_values1.append(mode_value)

    
    
    # Plot the mean, median, and mode as a function of k (file index)
    plt.plot(range(1, len(sorted_cfile_files1) + 1), mean_values1, label='Mean', marker='o', color='red', linestyle='-', markersize=5)
    plt.plot(range(1, len(sorted_cfile_files1) + 1), median_values1, label='Median', marker='o', color='green', linestyle='-', markersize=5)
    plt.plot(range(1, len(sorted_cfile_files1) + 1), mode_values1, label='Mode', marker='o', color='blue', linestyle='-', markersize=5)
    
    # for i, cfile in enumerate(sorted_cfile_files2, start=1):
    #     file_path = os.path.join(directory_path2, cfile)
        
    #     time, target_bin_values = process_iq_data(file_path)

    #     mean_value, median_value, mode_value = stat_for_targeted(target_bin_values)

    #     mean_values2.append(mean_value)
    #     median_values2.append(median_value)
    #     mode_values2.append(mode_value)

    # # Plot the mean, median, and mode as a function of k (file index)
    # # plt.plot(range(1, len(sorted_cfile_files2) + 1), mean_values2, label='Mean with object2', marker='o', color='green', linestyle='-', markersize=5)
    # plt.plot(range(1, len(sorted_cfile_files2) + 1), median_values2, label='Median without object', marker='o', color='green', linestyle='-', markersize=5)
    # # plt.plot(range(1, len(sorted_cfile_files2) + 1), mode_values2, label='Mode with object2', marker='o', color='green', linestyle='-', markersize=5)
    
    
    # for i, cfile in enumerate(sorted_cfile_files3, start=1):
    #     file_path = os.path.join(directory_path3, cfile)
        
    #     time, target_bin_values = process_iq_data(file_path)

    #     mean_value, median_value, mode_value = stat_for_targeted(target_bin_values)

    #     mean_values3.append(mean_value)
    #     median_values3.append(median_value)
    #     mode_values3.append(mode_value)

    

    # # Plot the mean, median, and mode as a function of k (file index)
    # # plt.plot(range(1, len(sorted_cfile_files3) + 1), mean_values3, label='Mean without object', marker='o', color='blue', linestyle='-', markersize=5)
    # plt.plot(range(1, len(sorted_cfile_files3) + 1), median_values3, label='Median turned off', marker='o', color='blue', linestyle='-', markersize=5)
    # # plt.plot(range(1, len(sorted_cfile_files3) + 1), mode_values3, label='Mode without object', marker='o', color='blue', linestyle='-', markersize=5)
    
    print("median values with : ", median_values1)
    print("median values without : ", median_values2)
    print("median values turned off : ", median_values3)

    # Add labels and title
    plt.xlabel('Distance (m)', fontsize=12)
    xticks = plt.gca().get_xticks()
    plt.xticks(xticks, [0.3 * int(x) for x in xticks])
    # plt.xticks(xticks, [int(x)/2 for x in xticks])
    plt.ylabel('Power (dBm)', fontsize=12)
    # plt.title(f'Variation of Median for {target_freq / 1e6} MHz  with Distance in meters with and without object', fontsize=14)
    plt.title(f'Variation of Mean, Median and Mode for {target_freq / 1e6} MHz  when the object is at 0.3m from the signal source', fontsize=14)
    plt.legend()
    plt.grid(True)

    # Save the variation plot
    # directory_path = "/media/oshani/Shared/UBUNTU/EMforTomography/30s_794_nirasha_researchlab" 
    # variation_output_path = os.path.join(directory_path, "basedonmode.png")
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


cfile_files = [f for f in os.listdir(directory_path1) if f.endswith('.cfile')]
sorted_cfile_files1 = sorted(cfile_files, key=lambda x: float(x.split('_')[1].split('f')[0]))
print(sorted_cfile_files1)

# cfile_files = [f for f in os.listdir(directory_path2) if f.endswith('.cfile')]
# sorted_cfile_files2 = sorted(cfile_files, key=lambda x: float(x.split('_')[1].split('m')[0]))
# print(sorted_cfile_files2)

# cfile_files = [f for f in os.listdir(directory_path3) if f.endswith('.cfile')]
# sorted_cfile_files3 = sorted(cfile_files, key=lambda x: float(x.split('_')[1].split('m')[0]))
# print(sorted_cfile_files3)

# allin1plot(sorted_cfile_files)
# plot_for_eachfile(sorted_cfile_files)
all_stat(sorted_cfile_files1)

