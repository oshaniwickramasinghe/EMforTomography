# import matplotlib.pyplot as plt
# import os
# import numpy as np
# import mmap
# from scipy import stats
# from scipy.stats import mode
# from scipy.signal import spectrogram

# # File name
# em_data = "/media/oshani/Shared/UBUNTU/EMforTomography/sensorlogs/compare/60s_1f_794_idle.cfile"
# sampling_frequency = 20e6  
# center_frequency = 794e6  
# target_freq = 792e6

# sensor_data = "/media/oshani/Shared/UBUNTU/EMforTomography/sensorlogs/compare/idle_sensor_log.txt"


# def read_iq_data(file_path):
#     print(f"Reading IQ data from: {file_path}")
#     with open(file_path, "rb") as f:
#         with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
#             file_size = mm.size()
#             num_samples = file_size // (2 * np.dtype(np.float32).itemsize)
#             raw_data = np.frombuffer(mm, dtype=np.float32).copy()
#             iq_data = raw_data[0::2] + 1j * raw_data[1::2]
#     print("Finished reading IQ data.")
#     return iq_data

# def compute_spectrogram(iq_segment):
#     """
#     Computes the spectrogram and filters the target frequency.
#     """
#     print("Computing spectrogram...")
#     frequencies, times, Sxx = spectrogram(
#         iq_segment,
#         fs=sampling_frequency,
#         window='hann',
#         nperseg=1024,
#         noverlap=512,
#         nfft=2048,
#         scaling='density'
#     )

#     frequencies_shifted = frequencies + (center_frequency - sampling_frequency / 2)

#     target_index = np.abs(frequencies_shifted - target_freq).argmin()
#     print(target_index)

#     print("Spectrogram computation complete.")

#     return times, 10 * np.log10(Sxx[target_index, :])  # Convert power to dB


# def stat_for_targeted(target_bin_values):
#     #see more
#     no_of_bins  = int(np.sqrt(len(target_bin_values)))
#     print(no_of_bins)

#     counts, bin_edges = np.histogram(target_bin_values, bins=no_of_bins)

#     # Calculate mean from histogram bin centers and counts (weighted mean)
#     bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2  # Calculate the bin centers
#     mean_value = np.sum(bin_centers * counts) / np.sum(counts)  # Weighted mean

#     # Calculate median from cumulative distribution
#     cumulative_counts = np.cumsum(counts)  # Cumulative sum of counts
#     median_bin_index = np.searchsorted(cumulative_counts, cumulative_counts[-1] / 2)  # Find the bin with cumulative count > 50%
#     median_value = bin_centers[median_bin_index]

#     # Calculate mode as the bin center with the highest count
#     mode_value = bin_centers[np.argmax(counts)]

#     return mean_value, median_value, mode_value


# #modifed to claculate mmm values of each segment
# def process_iq_data_over1file(file_path, interval_duration=1):
#     """
#     Processes IQ data from a file, extracting and plotting spectrogram segments.
#     """
#     iq_data = read_iq_data(file_path)
#     total_samples = len(iq_data)
#     samples_per_interval = int(sampling_frequency * interval_duration)
#     print("done reading")

#     interval_index = 0
#     temp_target_bin_segments=np.array([])
#     temp_time=np.array([])

#     mean_values = []
#     median_values = []
#     mode_values = []

#     while interval_index * samples_per_interval < total_samples:
#         start_sample = interval_index * samples_per_interval
#         end_sample = start_sample + samples_per_interval

#         # Extract segment and plot spectrogram
#         iq_segment = iq_data[start_sample:end_sample]
#         target_time, target_bin_segments=compute_spectrogram(iq_segment)

#         mean_value, median_value, mode_value = stat_for_targeted(target_bin_segments)

#         mean_values.append(mean_value)
#         median_values.append(median_value)
#         mode_values.append(mode_value)
        
#         # Concatenate the computed segments to the temporary arrays
#         temp_target_bin_segments = np.concatenate((temp_target_bin_segments, target_bin_segments))
#         temp_time = np.concatenate((temp_time, target_time))


#         interval_index += 1

#     # return temp_time,temp_target_bin_segments
#     return mean_values, median_values, mode_values


# def all_stat_for_every_second(file_path, label,color):

#     mean_values , median_values, mode_values = process_iq_data_over1file(file_path)

#     # Plot Mean values for the file
#     # plt.figure(figsize=(10, 5))
#     # plt.plot(range(1, len(mean_values) + 1), mean_values, label='Mean', marker='o', color='red', linestyle='-')
#     plt.plot(range(1, len(median_values) + 1), median_values, label=label, marker='o', color=color, linestyle='-')
#     # plt.plot(range(1, len(mode_values) + 1), mode_values, label='Mode', marker='o', color='blue', linestyle='-')

#     plt.xlabel('Time Interval (Seconds)', fontsize=12)
#     plt.ylabel('Power (dB)', fontsize=12)
#     plt.title(f'Power Variation and sensor data for idle state', fontsize=14)
#     plt.grid(True)
#     # plt.savefig(os.path.join(directory_path, f"mmmeverysecond_{cfile}.png"))
#     plt.legend()
#     # plt.show()

# # Initialize lists to store data
# times = []
# vcore_values = []
# avcc_values = []

# # Read the file
# with open(sensor_data, "r") as file:
#     lines = file.readlines()

#     # Skip the header line
#     for line in lines[1:]:
#         try:
#             # Split the line into columns
#             time, vcore, three_point_three, avcc = line.strip().split(", ")
#             times.append(time)  # Add time
#             vcore_values.append(float(vcore))  # Add Vcore value
#             avcc_values.append(float(avcc))   # Add AVCC value
#         except ValueError:
#             # Skip lines with errors (e.g., malformed lines)
#             continue

# # Plot the data
# plt.figure(figsize=(12, 6))

# # Plot Vcore
# # plt.plot(times, vcore_values, marker="o", label="Vcore (mV)", color="blue")

# # Plot AVCC
# plt.plot(times, avcc_values, marker="x", label="AVCC (V)", color="green")

# all_stat_for_every_second(em_data,label="EM data",color="green")


# # Format the graph
# plt.xticks(rotation=45, fontsize=5.5)  # Rotate and set font size for x-axis
# plt.xlabel("Time")
# plt.ylabel("Voltage")
# plt.title("AVCC Over Time when idle to compare with IQ median _ 825")
# plt.grid(True)
# plt.legend()  # Show the legend
# plt.tight_layout()

# # Show the plot
# plt.show()




import matplotlib.pyplot as plt
import os
import numpy as np
import mmap
from scipy.signal import spectrogram
import gc

# File paths
em_data = "/media/oshani/Shared/UBUNTU/EMforTomography/sensorlogs/compare/60s_1f_794_idle.cfile"
# em_data = "/media/oshani/Shared/UBUNTU/EMforTomography/30s_827_nirasha_researchlab/WN/30s_1f_827.cfile"
sensor_data = "/media/oshani/Shared/UBUNTU/EMforTomography/sensorlogs/compare/idle_sensor_log.txt"

# Constants
sampling_frequency = 20e6  
center_frequency = 794e6  
target_freq = 792e6  

def read_iq_data(file_path):
    """Reads complex IQ data from a binary file."""
    print(f"Reading IQ data from: {file_path}")
    with open(file_path, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            raw_data = np.frombuffer(mm, dtype=np.float32).copy()
            iq_data = raw_data[0::2] + 1j * raw_data[1::2]

    print(f"Done reading IQ data from: {file_path}")
    return iq_data


def read_partial_iq_data(file_path, start_percent=0, end_percent=30):
    """Reads a portion of the IQ data from a binary file."""
    print(f"Reading {start_percent}% to {end_percent}% of IQ data from: {file_path}")
    
    with open(file_path, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            file_size = mm.size()
            total_samples = file_size // (2 * np.dtype(np.float32).itemsize)

            start_sample = int((start_percent / 100) * total_samples)
            end_sample = int((end_percent / 100) * total_samples)

            raw_data = np.frombuffer(mm, dtype=np.float32, count=(end_sample - start_sample) * 2, offset=start_sample * 2 * np.dtype(np.float32).itemsize).copy()
            iq_data = raw_data[0::2] + 1j * raw_data[1::2]

    print(f"Done reading {start_percent}% to {end_percent}% of IQ data.")
    return iq_data

def compute_spectrogram(iq_segment):
    """Computes the spectrogram and extracts power at the target frequency."""
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
    return times, 10 * np.log10(Sxx[target_index, :])

# def process_iq_data(file_path, interval_duration=1):
#     """Processes IQ data in intervals and extracts median power values."""
#     iq_data = read_iq_data(file_path)
#     samples_per_interval = int(sampling_frequency * interval_duration)
#     total_samples = len(iq_data)
    
#     median_values = []
#     for i in range(0, total_samples, samples_per_interval):
#         iq_segment = iq_data[i : i + samples_per_interval]
#         _, target_bin_segments = compute_spectrogram(iq_segment)
#         median_values.append(np.median(target_bin_segments))
    
#     return median_values

def process_iq_data(iq_data, interval_duration=1):
    """Processes IQ data in intervals and extracts median power values."""
    # iq_data = read_iq_data(file_path)
    samples_per_interval = int(sampling_frequency * interval_duration)
    total_samples = len(iq_data)
    
    median_values = []
    for i in range(0, total_samples, samples_per_interval):
        iq_segment = iq_data[i : i + samples_per_interval]
        _, target_bin_segments = compute_spectrogram(iq_segment)
        median_values.append(np.median(target_bin_segments))
    
    return median_values

def read_sensor_data(sensor_file):
    """Reads AVCC and Vcore values from a sensor log file."""
    times, vcore_values, avcc_values = [], [], []
    with open(sensor_file, "r") as file:
        for line in file.readlines()[1:]:  # Skip header
            try:
                time, vcore, _, avcc = line.strip().split(", ")
                times.append(time)
                vcore_values.append(float(vcore))
                avcc_values.append(float(avcc))
            except ValueError:
                continue
    return times, vcore_values, avcc_values

def plot_results(times, vcore_values, avcc_values, median_values):
    """Plots AVCC, Vcore, and EM data medians in subplots."""
    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    
    axs[0].plot(times, avcc_values, marker='o', color='green', label='AVCC (V)')
    axs[0].set_ylabel("Voltage (V)")
    axs[0].set_title("AVCC Over Time")
    axs[0].grid(True)
    axs[0].legend()

    axs[1].plot(range(1, len(median_values) + 1), median_values, marker='o', color='red', label='EM Median Power (dB)')
    axs[1].set_ylabel("Power (dB)")
    axs[1].set_title("Median EM Power Over Time")
    axs[1].grid(True)
    axs[1].legend()
    
    axs[2].plot(times, vcore_values, marker='o', color='blue', label='Vcore (mV)')
    axs[2].set_xlabel("Time Interval (Seconds)")
    axs[2].set_ylabel("Voltage (mV)")
    axs[2].set_title("Vcore Over Time")
    axs[2].grid(True)
    axs[2].legend()
   
    
    plt.xticks(rotation=45, fontsize=6)
    plt.tight_layout()
    plt.show()

# Execute processing and plotting
times, vcore_values, avcc_values = read_sensor_data(sensor_data)

# Process first segment (0-30%)
iq_data_part1 = read_partial_iq_data(em_data, start_percent=0, end_percent=30)
median_values_part1 = process_iq_data(iq_data_part1)
median_values = median_values_part1

# Free memory
del iq_data_part1, median_values_part1
gc.collect()

# Process second segment (30-60%)
iq_data_part2 = read_partial_iq_data(em_data, start_percent=30, end_percent=60)
median_values_part2 = process_iq_data(iq_data_part2)
median_values += median_values_part2

# Free memory
del iq_data_part2, median_values_part2
gc.collect()

# Process third segment (60-90%)
iq_data_part3 = read_partial_iq_data(em_data, start_percent=60, end_percent=90)
median_values_part3 = process_iq_data(iq_data_part3)
median_values += median_values_part3

# Free memory
del iq_data_part3, median_values_part3
gc.collect()

# Process fourth segment (90-100%)
iq_data_part4 = read_partial_iq_data(em_data, start_percent=90, end_percent=100)
median_values_part4 = process_iq_data(iq_data_part4)
median_values += median_values_part4

# Free memory
del iq_data_part4, median_values_part4
gc.collect()

# Plot the combined results
plot_results(times, vcore_values, avcc_values, median_values)
