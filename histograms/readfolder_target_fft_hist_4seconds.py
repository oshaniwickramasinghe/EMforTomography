#process files for each second
#create mmm plot for each second of a file
#create histogram for each second of a file
import os
import numpy as np
import mmap
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import mode

# Directory path containing .cfile files
# directory_path = "/media/oshani/Shared/UBUNTU/EMforTomography/893/no_object"

# # Parameters  Hz
# sampling_frequency = 20e6  
# center_frequency = 893e6  
# target_freq = 891e6


# directory_path = "/media/oshani/Shared/UBUNTU/EMforTomography/827/no_object"
# sampling_frequency = 20e6  
# center_frequency = 827e6  
# target_freq = 825e6

directory_path = "/media/oshani/Shared/UBUNTU/EMforTomography/794/no_object"
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


def calculate_spectrogram(samples):
    print(f"Calculating spectrogram")
    fft_size = 1024
    num_rows = len(samples) // fft_size
    spectrogram = np.zeros((num_rows, fft_size))

    # Perform FFT and calculate the spectrogram
    for i in range(num_rows):
        spectrogram[i, :] = 10 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(samples[i * fft_size:(i+1) * fft_size]))) ** 2)

    # Calculate the frequency resolution
    freq_resolution = sampling_frequency / fft_size  # Hz per bin
    target_bin = int((target_freq - center_frequency) / freq_resolution + fft_size // 2)  # Convert to bin index

    # Extract the power values for the targeted MHz bin
    target_bin_values = spectrogram[:, target_bin]

    print(f"Calculation done")
    return target_bin_values
  

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




# Function to process IQ data and plot statistics for every second of each file
def all_stat_for_every_second(cfile_files):

    # cfile_files=['794_8t_null.cfile']
    
    for i, cfile in enumerate(cfile_files, start=1):
        file_path = os.path.join(directory_path, cfile)
        print(f"Processing file: {cfile}")

        # Read IQ data
        iq_data = read_iq_data(file_path)
        total_samples = len(iq_data)
        interval_duration = 1  # second
        samples_per_interval = int(sampling_frequency * interval_duration)

        mean_values = []
        median_values = []
        mode_values = []

        interval_index = 0
        while True:
            start_sample = interval_index * samples_per_interval
            end_sample = start_sample + samples_per_interval

            if end_sample > total_samples:
                print(f"End of file reached at interval {interval_index}. Exiting loop.")
                break

            # Extract the segment
            iq_segment = iq_data[start_sample:end_sample]

            # Calculate statistics for the target frequency
            target_bin_values = calculate_spectrogram(iq_segment)

            mean_value, median_value, mode_value = stat_for_targeted(target_bin_values)

            mean_values.append(mean_value)
            median_values.append(median_value)
            mode_values.append(mode_value)

            interval_index += 1

        # Plot Mean values for the file
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(mean_values) + 1), mean_values, label='Mean', marker='o', color='red', linestyle='-')
        plt.plot(range(1, len(median_values) + 1), median_values, label='Median', marker='o', color='green', linestyle='-')
        plt.plot(range(1, len(mode_values) + 1), mode_values, label='Mode', marker='o', color='blue', linestyle='-')

        plt.xlabel('Time Interval (Seconds)', fontsize=12)
        plt.ylabel('Power (dB)', fontsize=12)
        plt.title(f'Power Variation for {cfile}', fontsize=14)
        plt.grid(True)
        plt.savefig(os.path.join(directory_path, f"mmmeverysecond_{cfile}.png"))
        plt.legend()
        plt.show()

        # # Plot Median values for the file
        # plt.figure(figsize=(10, 5))
        # plt.plot(range(1, len(median_values) + 1), median_values, label='Median', marker='o', color='green', linestyle='-')
        # plt.xlabel('Time Interval (Seconds)', fontsize=12)
        # plt.ylabel('Median Power (dB)', fontsize=12)
        # plt.title(f'Median Power Variation for {cfile}', fontsize=14)
        # plt.grid(True)
        # plt.savefig(os.path.join(directory_path, f"median_{cfile}.png"))
        # plt.show()

        # # Plot Mode values for the file
        # plt.figure(figsize=(10, 5))
        # plt.plot(range(1, len(mode_values) + 1), mode_values, label='Mode', marker='o', color='blue', linestyle='-')
        # plt.xlabel('Time Interval (Seconds)', fontsize=12)
        # plt.ylabel('Mode Power (dB)', fontsize=12)
        # plt.title(f'Mode Power Variation for {cfile}', fontsize=14)
        # plt.grid(True)
        # plt.savefig(os.path.join(directory_path, f"mode_{cfile}.png"))
        # plt.show()
       


# Function to process IQ data and plot statistics for every second of each file
def for_every_second(cfile_files):

    cfile_files=['794_9t_null.cfile']
    
    for i, cfile in enumerate(cfile_files, start=1):
        file_path = os.path.join(directory_path, cfile)
        print(f"Processing file: {cfile}")

        # Read IQ data
        iq_data = read_iq_data(file_path)
        total_samples = len(iq_data)
        interval_duration = 0.5  # second
        samples_per_interval = int(sampling_frequency * interval_duration)

        plt.figure(figsize=(20, 12))
        interval_index = 0
        while True:
            start_sample = interval_index * samples_per_interval
            end_sample = start_sample + samples_per_interval

            if end_sample > total_samples:
                print(f"End of file reached at interval {interval_index}. Exiting loop.")
                break

            # Extract the segment
            iq_segment = iq_data[start_sample:end_sample]

            # Calculate statistics for the target frequency
            target_bin_values = calculate_spectrogram(iq_segment)

            # plt.figure(figsize=(20, 12))

            plot_histogram_for_targeted(target_bin_values, k=interval_index, label=interval_index, title=f"{cfile}  {interval_index}", alpha_mean=0.7, alpha_median=0.7, alpha_mode=0.7)

            interval_index += 1

            # plt.show()
        plt.show()

    


    
cfile_files = [f for f in os.listdir(directory_path) if f.endswith('.cfile')]
print(cfile_files)

sorted_cfile_files = sorted(cfile_files, key=lambda x: int(x.split('_')[1].split('t')[0]))

print(sorted_cfile_files)

all_stat_for_every_second(sorted_cfile_files)


