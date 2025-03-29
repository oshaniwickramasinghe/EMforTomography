import os
import numpy as np
import mmap
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import mode
from scipy.signal import spectrogram


# directory_path1 = "/media/oshani/Shared/UBUNTU/EMforTomography/waru/794/waruat1feet/use"
# directory_path2 = "/media/oshani/Shared/UBUNTU/EMforTomography/waru/827/waruat1.5feet"
# directory_path3 = "/media/oshani/Shared/UBUNTU/EMforTomography/waru/827/waruat2feet"
# directory_path4 = "/media/oshani/Shared/UBUNTU/EMforTomography/waru/827/waruat2.5feet"
# directory_path5 = "/media/oshani/Shared/UBUNTU/EMforTomography/waru/827/waruat3feet"

# directory_path6 = "/media/oshani/Shared/UBUNTU/EMforTomography/waru/794/withoutwaru/use"
# directory_path7 = "/media/oshani/Shared/UBUNTU/EMforTomography/waru/794/desktopoff/use"

# directory_path8 = "/media/oshani/Shared/UBUNTU/EMforTomography/waru/794/movingwaruantennaat7/nouse"
# directory_path9 = "/media/oshani/Shared/UBUNTU/EMforTomography/waru/794/movingwaruantennaat7"
# directory_path10 = "/media/oshani/Shared/UBUNTU/EMforTomography/waru/794/desktopoff/use"

directory_path8 = "/media/oshani/Shared/UBUNTU/EMforTomography/waru/794/waruat1feet/use"
directory_path9 = "/media/oshani/Shared/UBUNTU/EMforTomography/waru/794/withoutwaru/use"


# directory_path11 = "/media/oshani/Shared/UBUNTU/EMforTomography/waru/794/movingwaruantennaat9/nouse"
# directory_path12 = "/media/oshani/Shared/UBUNTU/EMforTomography/waru/794/movingwaruantennaat9"
# directory_path13 = "/media/oshani/Shared/UBUNTU/EMforTomography/waru/794/desktopoff/use"

sampling_frequency = 20e6  
center_frequency = 794e6  
target_freq = 792e6


def read_iq_data(file_path):
    print(f"Reading IQ data from: {file_path}")
    with open(file_path, "rb") as f:
        print("111")
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            file_size = mm.size()
            num_samples = file_size // (2 * np.dtype(np.float32).itemsize)
            raw_data = np.frombuffer(mm, dtype=np.float32).copy()
            iq_data = raw_data[0::2] + 1j * raw_data[1::2]
            print(iq_data)
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
    plt.ylabel("Signal Strength [dBm]")
    plt.title(f"Signal Strength at {target_freq / 1e6} MHz from {k*2} feet distance")
    # plt.title(f"Signal Strength at {target_freq / 1e6} MHz ")
    plt.grid(True)
    # plt.legend()

    
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


def all_stat(sorted_cfile_files, directory_path, label, color, cache, output_file="median_values.txt"):

    median_values1 = []

    with open(output_file, 'a') as file:  # Open the file in append mode
        for cfile in sorted_cfile_files:
            file_path = os.path.join(directory_path, cfile)

            if file_path in cache:
                median_value = cache[file_path]
            else:
                time, target_bin_values = process_iq_data(file_path)
                mean_value, median_value, mode_value = stat_for_targeted(target_bin_values)
                cache[file_path] = median_value  # Store for reuse
            
            median_values1.append(median_value)
            file.write(f"{file_path}    {median_value}\n")  # Append each median value to the file

    x_values = range(1, len(median_values1) + 1)

    # Add data labels next to each point
    for x, y in zip(x_values, median_values1):
        plt.text(x, y, f'{y:.2f}', fontsize=9, ha='right', va='bottom', color=color)

    plt.plot(x_values, median_values1, label=label, marker='o', color=color, linestyle='-', markersize=5)
    print(f"Median values for {label}: ", median_values1)

        # Add labels and title
    plt.xlabel('Distance (m)', fontsize=12)
    xticks = plt.gca().get_xticks()
    plt.xticks(xticks, [0.3*int(x) for x in xticks])
    # plt.xticks(xticks, [int(x)/2 for x in xticks])
    plt.ylabel('Power (dBm)', fontsize=12)
    plt.title(f'Variation of Median for {target_freq / 1e6} MHz  with Distance in centimeters with and without the object at ground', fontsize=14)
    # plt.title(f'Variation of Median for {target_freq / 1e6} MHz  when the object is moving away from source', fontsize=14)
    # plt.legend()
    plt.grid(True)
    # plt.show()


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


# cfile_files = [f for f in os.listdir(directory_path1) if f.endswith('.cfile')]
# sorted_cfile_files1 = sorted(cfile_files, key=lambda x: float(x.split('_')[1].split('f')[0]))
# print(sorted_cfile_files1)

# cfile_files = [f for f in os.listdir(directory_path2) if f.endswith('.cfile')]
# sorted_cfile_files2 = sorted(cfile_files, key=lambda x: float(x.split('_')[1].split('f')[0]))
# print(sorted_cfile_files2)

# cfile_files = [f for f in os.listdir(directory_path3) if f.endswith('.cfile')]
# sorted_cfile_files3 = sorted(cfile_files, key=lambda x: float(x.split('_')[1].split('f')[0]))
# print(sorted_cfile_files3)

# cfile_files = [f for f in os.listdir(directory_path4) if f.endswith('.cfile')]
# sorted_cfile_files4 = sorted(cfile_files, key=lambda x: float(x.split('_')[1].split('f')[0]))
# print(sorted_cfile_files4)

# cfile_files = [f for f in os.listdir(directory_path5) if f.endswith('.cfile')]
# sorted_cfile_files5 = sorted(cfile_files, key=lambda x: float(x.split('_')[1].split('f')[0]))
# print(sorted_cfile_files5)

# cfile_files = [f for f in os.listdir(directory_path6) if f.endswith('.cfile')]
# sorted_cfile_files6 = sorted(cfile_files, key=lambda x: float(x.split('_')[1].split('f')[0]))
# print(sorted_cfile_files6)

# cfile_files = [f for f in os.listdir(directory_path7) if f.endswith('.cfile')]
# sorted_cfile_files7 = sorted(cfile_files, key=lambda x: float(x.split('_')[1].split('f')[0]))
# print(sorted_cfile_files7)

cfile_files = [f for f in os.listdir(directory_path8) if f.endswith('.cfile')]
sorted_cfile_files8 = sorted(cfile_files, key=lambda x: float(x.split('_')[1].split('f')[0]))
print(sorted_cfile_files8)

cfile_files = [f for f in os.listdir(directory_path9) if f.endswith('.cfile')]
sorted_cfile_files9 = sorted(cfile_files, key=lambda x: float(x.split('_')[1].split('f')[0]))
print(sorted_cfile_files9)

# cfile_files = [f for f in os.listdir(directory_path10) if f.endswith('.cfile')]
# sorted_cfile_files10 = sorted(cfile_files, key=lambda x: float(x.split('_')[1].split('f')[0]))
# print(sorted_cfile_files10)

# cfile_files = [f for f in os.listdir(directory_path11) if f.endswith('.cfile')]
# sorted_cfile_files11 = sorted(cfile_files, key=lambda x: float(x.split('_')[1].split('f')[0]))
# print(sorted_cfile_files11)

# cfile_files = [f for f in os.listdir(directory_path12) if f.endswith('.cfile')]
# sorted_cfile_files12 = sorted(cfile_files, key=lambda x: float(x.split('_')[1].split('f')[0]))
# print(sorted_cfile_files12)

# cfile_files = [f for f in os.listdir(directory_path13) if f.endswith('.cfile')]
# sorted_cfile_files13 = sorted(cfile_files, key=lambda x: float(x.split('_')[1].split('f')[0]))
# print(sorted_cfile_files13)

plt.figure(figsize=(12, 6))

cache = {}  # Dictionary to store processed file results
output_file = "median_values.txt"


# all_stat(sorted_cfile_files1, directory_path1, label="object at 30cm", color ="green", cache=cache)
# all_stat(sorted_cfile_files2, directory_path2, label="object at 45cm", color ="blue", cache=cache)
# all_stat(sorted_cfile_files3, directory_path3, label="object at 60cm", color ="purple", cache=cache)
# all_stat(sorted_cfile_files4, directory_path4, label="object at 75cm", color ="pink", cache=cache)
# all_stat(sorted_cfile_files5, directory_path5, label="object at 90cm", color ="yellow", cache=cache)

# all_stat(sorted_cfile_files6, directory_path6, label="without object", color ="red", cache=cache)
# all_stat(sorted_cfile_files7, directory_path7, label="desktop off", color ="black", cache=cache)


# sorted_cfile_files10 = ['30s_7f.cfile','30s_7f.cfile','30s_7f.cfile','30s_7f.cfile','30s_7f.cfile','30s_7f.cfile','30s_7f.cfile']
# all_stat(sorted_cfile_files10, directory_path10, label="desktop off", color ="black", cache=cache)
# sorted_cfile_files8 = ['30s_7f_WOWend.cfile','30s_7f_WOWend.cfile','30s_7f_WOWend.cfile','30s_7f_WOWend.cfile','30s_7f_WOWend.cfile','30s_7f_WOWend.cfile','30s_7f_WOWend.cfile']
all_stat(sorted_cfile_files8, directory_path8, label="with object", color ="red", cache=cache)
all_stat(sorted_cfile_files9, directory_path9, label="without object", color ="green", cache=cache)


# sorted_cfile_files13 = ['30s_9f.cfile','30s_9f.cfile','30s_9f.cfile','30s_9f.cfile','30s_9f.cfile','30s_9f.cfile','30s_9f.cfile']
# all_stat(sorted_cfile_files13, directory_path13, label="desktop off", color ="black", cache=cache)
# sorted_cfile_files11 = ['30s_9f_WOWend.cfile','30s_9f_WOWend.cfile','30s_9f_WOWend.cfile','30s_9f_WOWend.cfile','30s_9f_WOWend.cfile','30s_9f_WOWend.cfile','30s_9f_WOWend.cfile']
# all_stat(sorted_cfile_files11, directory_path11, label="without object", color ="red", cache=cache)
# all_stat(sorted_cfile_files12, directory_path12, label="moving object", color ="blue", cache=cache)


plt.legend()
plt.show()