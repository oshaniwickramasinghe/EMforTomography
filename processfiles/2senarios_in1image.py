import os
import numpy as np
import mmap
import matplotlib.pyplot as plt
from scipy.signal import spectrogram


directory_path1 = "/media/oshani/Shared/UBUNTU/EMforTomography/30s_794_nirasha_researchlab/withNirasha"
directory_path2 = "/media/oshani/Shared/UBUNTU/EMforTomography/30s_794_nirasha_researchlab/withoutNirasha"

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


def plot_spectrogram_for_targeted(time, target_bin_values, label, color):
   
    plt.plot(time, target_bin_values, linewidth=0.5, label=label, color=color)    
    plt.xlabel("Time [s]")
    plt.ylabel("Signal Strength [dB]")
    plt.title(f"Signal Strength at {target_freq / 1e6} MHz")
    plt.grid(True)
    plt.legend()

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

def plot_stat_for_targeted(target_bin_values, k, label, title, alpha_mean, alpha_median, alpha_mode):
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


def allin1plot(cfile_files, directory_path):

    # plt.figure(figsize=(20, 12))

    # Loop through all the files and read the IQ data
    for i, cfile in enumerate(cfile_files, start=1):
        file_path = os.path.join(directory_path, cfile)

        time, target_bin_values = process_iq_data(file_path)


        # plot_histogram_for_targeted(target_bin_values, i, label=f"{i*2} feet")  
        plot_stat_for_targeted(target_bin_values, i, label=f"{i*2} feet", title=f"Histogram of Power Values at {target_freq / 1e6} MHz", alpha_mean=0.7, alpha_median=0.7, alpha_mode=0.7)
        # plot_spectrogram_for_targeted(time, target_bin_values, label=f"{i*2} feet", color="blue")          
        
        print("done  ", i)
        
    # Save the variation plot
    variation_output_path = os.path.join(directory_path, "histogram.png")
    plt.savefig(variation_output_path)
    # plt.show() 
    

cfile_files1 = [f for f in os.listdir(directory_path1) if f.endswith('.cfile')]
print(cfile_files1)
sorted_cfile_files1= sorted(cfile_files1, key=lambda x: int(x.split('_')[1].split('f')[0]))
print(sorted_cfile_files1)

cfile_files2 = [f for f in os.listdir(directory_path2) if f.endswith('.cfile')]
print(cfile_files2)
sorted_cfile_files2= sorted(cfile_files2, key=lambda x: int(x.split('_')[1].split('f')[0]))
print(sorted_cfile_files2)

plt.figure(figsize=(20, 12))
allin1plot(sorted_cfile_files1, directory_path1)
allin1plot(sorted_cfile_files2, directory_path2)

plt.show() 