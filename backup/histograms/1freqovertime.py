# plo spectograms for 2 different files one over another. can be used to see how signal strength change with time
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import mmap

# File paths
iq_data_file1 = "/media/oshani/Shared/UBUNTU/EMforTomography/893/dasun/893_3t.cfile"
iq_data_file2 = "/media/oshani/Shared/UBUNTU/EMforTomography/893/no_object/893_3t_null.cfile"

# Parameters
sampling_frequency = 20e6  # Hz
center_frequency = 893e6  # Hz
# freq_target = 800e6  # Hz (Target frequency to analyze)

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


    return times, 10 * np.log10(Sxx[target_index, :] + 1e-12)  # Convert power to dB


def plot_spectrogram(time, target_bin_values, label, color):

    print(time)
    print(f"Size of times: {time.shape if isinstance(time, np.ndarray) else len(times)}")

   
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


time1,target_bin1=process_iq_data(iq_data_file1)
time2,target_bin2=process_iq_data(iq_data_file2)

plt.figure(figsize=(20, 10))

plot_spectrogram(time1,target_bin1, label="with", color="blue")
plot_spectrogram(time2,target_bin2, label="without", color="orange")

plt.show()

#    # Read IQ data
# iq_data1 = read_iq_data(iq_data_file1)
# iq_data2 = read_iq_data(iq_data_file2)

# interval_duration = 1  # second
# samples_per_interval = int(sampling_frequency * interval_duration)

# # Initialize the figure
# plt.figure(figsize=(20, 10))

# # Process first IQ data file (with label 'with')
# total_samples = len(iq_data1)
# interval_index = 0
# while True:
#     start_sample = interval_index * samples_per_interval
#     end_sample = start_sample + samples_per_interval
    
#     if end_sample > total_samples:
#         print(f"No more data to process at interval {interval_index}. Exiting loop.")
#         break
    
#     # Extract the segment
#     iq_segment = iq_data1[start_sample:end_sample]

#     # Generate spectrogram and append to the figure
#     plot_spectrogram(iq_segment, label="with", interval_index=interval_index, color="blue")
   
#     interval_index += 1



# # Process second IQ data file (with label 'without')
# total_samples = len(iq_data2)
# interval_index = 0
# while True:
#     start_sample = interval_index * samples_per_interval
#     end_sample = start_sample + samples_per_interval
    
#     if end_sample > total_samples:
#         print(f"No more data to process at interval {interval_index}. Exiting loop.")
#         break
    
#     # Extract the segment
#     iq_segment = iq_data2[start_sample:end_sample]

#     # Generate spectrogram and append to the figure
#     plot_spectrogram(iq_segment, label="without", interval_index=interval_index, color="orange")
       
#     interval_index += 1

# # Display the figure
# plt.show()