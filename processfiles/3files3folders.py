import os
import numpy as np
import mmap
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

# # # Directory path containing .cfile files
# directory_path1 = "/media/oshani/Shared/UBUNTU/EMforTomography/893/dasun"
# directory_path2 = "/media/oshani/Shared/UBUNTU/EMforTomography/893/samindu"
# directory_path3 = "/media/oshani/Shared/UBUNTU/EMforTomography/893/no_object"
# # Parameters  Hz
# sampling_frequency = 20e6  
# center_frequency = 893e6  
# target_freq = 891e6


directory_path1 = "/media/oshani/Shared/UBUNTU/EMforTomography/827/dasun"
directory_path2 = "/media/oshani/Shared/UBUNTU/EMforTomography/827/samindu"
directory_path3 = "/media/oshani/Shared/UBUNTU/EMforTomography/827/no_object"
sampling_frequency = 20e6  
center_frequency = 827e6  
target_freq = 825e6

# directory_path1 = "/media/oshani/Shared/UBUNTU/EMforTomography/794/dasun"
# directory_path2 = "/media/oshani/Shared/UBUNTU/EMforTomography/794/samindu"
# directory_path3 = "/media/oshani/Shared/UBUNTU/EMforTomography/794/no_object"
# sampling_frequency = 20e6  
# center_frequency = 794e6  
# target_freq = 792e6

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


def all_stat(directory_path1, mean_fig,median_fig,mode_fig,mean_label,median_label,mode_label,color, alpha_mean=0.7, alpha_median=0.7, alpha_mode=0.7):

    mean_values1 = []
    median_values1 = []
    mode_values1 = []

    cfile_files = [f for f in os.listdir(directory_path1) if f.endswith('.cfile')]
    sorted_cfile_files1 = sorted(cfile_files, key=lambda x: int(x.split('_')[1].split('t')[0]))
    print(sorted_cfile_files1)


    # Now, create a separate plot showing the variation of mean, median, and mode with respect to k (i.e., the file index)
    # plt.figure(figsize=(12, 6))

    for i, cfile in enumerate(sorted_cfile_files1, start=1):
        file_path = os.path.join(directory_path1, cfile)
        
        time, target_bin_values = process_iq_data(file_path)

        mean_value, median_value, mode_value = stat_for_targeted(target_bin_values)

        mean_values1.append(mean_value) 
        median_values1.append(median_value)
        mode_values1.append(mode_value)

    
    
    # Plot the mean, median, and mode as a function of k (file index)
    plt.figure(mean_fig.number)
    plt.plot(range(1, len(sorted_cfile_files1) + 1), mean_values1, alpha=alpha_mean, label=mean_label, marker='o', color=color, linestyle='-', markersize=5)
    plt.figure(median_fig.number)
    plt.plot(range(1, len(sorted_cfile_files1) + 1), median_values1, alpha=alpha_median, label=median_label, marker='o', color=color, linestyle='-', markersize=5)
    plt.figure(mode_fig.number)
    plt.plot(range(1, len(sorted_cfile_files1) + 1),  mode_values1, alpha=alpha_mode, label=mode_label, marker='o', color=color, linestyle='-', markersize=5)
    

    # Add labels, legends, and grids for each figure
    for fig, ylabel, title in zip([mean_fig, median_fig, mode_fig],
                                ['Mean (dB)', 'Median (dB)', 'Mode (dB)'],
                                ['Variation of Meanat 825MHZ', 'Variation of Median 825MHZ', 'Variation of Mode 825MHZ']):
        plt.figure(fig.number)
        plt.xlabel('Distance (feet)', fontsize=12)
        xticks = plt.gca().get_xticks()
        plt.xticks(xticks, [2 * int(x)+4 for x in xticks])
        plt.ylabel(ylabel, fontsize=12)
        plt.title(title, fontsize=14)
        plt.legend()
        plt.grid(True)

        # Save the variation plot
        directory_path = "/media/oshani/Shared/UBUNTU/EMforTomography/images" 
        variation_output_path = os.path.join(directory_path, f"825_{title}.png")
        # plt.savefig(variation_output_path)
        # plt.show()



    



mean_fig = plt.figure(figsize=(12, 6))
median_fig = plt.figure(figsize=(12, 6))
mode_fig = plt.figure(figsize=(12, 6))


# allin1plot(sorted_cfile_files)
# plot_for_eachfile(sorted_cfile_files)
all_stat(directory_path1,mean_fig,median_fig,mode_fig, mean_label='mean with object 1', median_label='median with obejct 1', mode_label='mode with object 1', color='lightcoral')
all_stat(directory_path2,mean_fig,median_fig,mode_fig, mean_label='mean with object 2', median_label='median with obejct 2', mode_label='mode with object 2', color='springgreen')
all_stat(directory_path3,mean_fig,median_fig,mode_fig, mean_label='mean without object', median_label='median without object', mode_label='mode without object', color='cornflowerblue')


plt.show()