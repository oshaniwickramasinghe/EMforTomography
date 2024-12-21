import os
import numpy as np
import mmap
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import mode

# Directory path containing .cfile files
directory_path = "/media/oshani/Shared/UBUNTU/EMforTomography/827/no_object"

# Parameters  Hz
sampling_frequency = 20e6  
center_frequency = 827e6  
target_freq = 825e6


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

# Plot the signal strength of targted frequency over time
def plot_spectrogram_for_targeted(target_bin_values, k, label):
    plt.plot(target_bin_values, linewidth=0.2, alpha=0.9, label=label)    
    plt.xlabel("Time [s]")
    plt.ylabel("Signal Strength [dB]")
    plt.title(f"Signal Strength at {target_freq / 1e6} MHz from {k*2} feet distance")
    plt.title(f"Signal Strength at {target_freq / 1e6} MHz ")
    plt.grid(True)
    plt.legend()

# Plot a histogram of the power values in the target bin
def plot_histogram_for_targeted(target_bin_values, k, label):
    no_of_bins  = int(np.sqrt(len(target_bin_values)))
    print(no_of_bins)

    plt.hist(target_bin_values, bins=no_of_bins, alpha=0.7, label=label)
    plt.xlabel("Power (dB)")
    plt.ylabel("Frequency")
    plt.title(f"Histogram of Power Values at {target_freq / 1e6} MHz from {k*2} feet distance")
    plt.title(f"Histogram of Power Values at {target_freq / 1e6} MHz")
    plt.grid(True)

    mean_value, median_value, mode_value = stat_for_targeted(target_bin_values)

    # Mark the mean, median, and mode on the histogram
    plt.axvline(mean_value, color='red', alpha=0.7, linestyle='dashed', linewidth=2, label=f"Mean: {mean_value:.2f} dB")
    plt.axvline(median_value, color='green', alpha=0.7, linestyle='dashed', linewidth=2, label=f"Median: {median_value:.2f} dB")
    plt.axvline(mode_value, color='blue', alpha=0.7, linestyle='dashed', linewidth=2, label=f"Mode: {mode_value:.2f} dB")

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


def all_stat(cfile_files):

    mean_values = []
    median_values = []
    mode_values = []

    for i, cfile in enumerate(cfile_files, start=1):
        file_path = os.path.join(directory_path, cfile)
        iq_data = read_iq_data(file_path)

        target_bin_values = calculate_spectrogram(iq_data)

        mean_value, median_value, mode_value = stat_for_targeted(target_bin_values)

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
    plt.ylabel('Power (dB)', fontsize=12)
    plt.title('Variation of Mean, Median, and Mode with Distance in feet', fontsize=14)
    plt.legend()
    plt.grid(True)

    # Save the variation plot
    variation_output_path = os.path.join(directory_path, "variation_plotnew.png")
    plt.savefig(variation_output_path)
    plt.show()

def allin1plot(cfile_files):

    plt.figure(figsize=(20, 12))

    # Loop through all the files and read the IQ data
    for i, cfile in enumerate(cfile_files, start=1):
        file_path = os.path.join(directory_path, cfile)
        iq_data = read_iq_data(file_path)

        target_bin_values = calculate_spectrogram(iq_data)

        plot_histogram_for_targeted(target_bin_values, i, label=f"{i*2} feet")  
        # plot_spectrogram_for_targeted(target_bin_values, i, label=f"{i*2} feet")          
        
        print("done  ", i)
        
    # Save the variation plot
    variation_output_path = os.path.join(directory_path, "variation_plotnew.png")
    plt.savefig(variation_output_path)
    plt.show() 



def plot_for_eachfile(cfile_files):

    # Loop through all the files and read the IQ data
    for i, cfile in enumerate(cfile_files, start=1):
        file_path = os.path.join(directory_path, cfile)
        iq_data = read_iq_data(file_path)

        target_bin_values = calculate_spectrogram(iq_data)
        
        plt.figure(figsize=(20, 12))

        plot_histogram_for_targeted(target_bin_values, i, label=f"{i*2} feet")  
        # plot_spectrogram_for_targeted(target_bin_values, i, label=f"{i*2} feet")  

        print("done  ", i)
        
        # Save the variation plot
        variation_output_path = os.path.join(directory_path, "variation_plotnew.png")
        plt.savefig(variation_output_path)
        plt.show()



cfile_files = [f for f in os.listdir(directory_path) if f.endswith('.cfile')]
print(cfile_files)

allin1plot(cfile_files)
plot_for_eachfile(cfile_files)
all_stat(cfile_files)

