# not working  for 891 frequency 
import os
import numpy as np
import mmap
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import mode
from scipy.signal import find_peaks

# Directory path containing .cfile files
directory_path = "/media/oshani/Shared/UBUNTU/EMforTomography/waru/794/waruat1feet/use"

# Parameters  Hz
sampling_frequency = 20e6  
center_frequency = 893e6  
target_freq = 891e6


# directory_path = "/media/oshani/Shared/UBUNTU/EMforTomography/827/samindu"
# sampling_frequency = 20e6  
# center_frequency = 827e6  
# target_freq = 825e6

# directory_path = "/media/oshani/Shared/UBUNTU/EMforTomography/794/samindu"
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
    plt.axvline(mean_value, color='red', alpha=alpha_mean, linestyle='dashed', linewidth=2, label=f"Mean: {mean_value:.2f} dB")
    plt.axvline(median_value, color='green', alpha=alpha_median, linestyle='dashed', linewidth=2, label=f"Median: {median_value:.2f} dB")
    plt.axvline(mode_value, color='blue', alpha=alpha_mode, linestyle='dashed', linewidth=2, label=f"Mode: {mode_value:.2f} dB")

    plt.legend()
    plt.show()

    

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

    print(cfile_files)
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
    xticks = plt.gca().get_xticks()
    plt.xticks(xticks, [2 * int(x)+4 for x in xticks])
    plt.ylabel('Power (dB)', fontsize=12)
    plt.title(f'Variation of Mean, Median, and Mode for {target_freq / 1e6} MHz  with Distance in feet with object', fontsize=14)
    plt.legend()
    plt.grid(True)

    # Save the variation plot
    variation_output_path = os.path.join(directory_path, "allstat.png")
    plt.savefig(variation_output_path)
    plt.show()

def allin1plot(cfile_files):

    plt.figure(figsize=(20, 12))

    # Loop through all the files and read the IQ data
    for i, cfile in enumerate(cfile_files, start=1):
        file_path = os.path.join(directory_path, cfile)
        iq_data = read_iq_data(file_path)

        no_of_bins = int(np.sqrt(len(target_bin_values)))

        # Generate histogram data
        counts, bin_edges = np.histogram(iq_data, bins=no_of_bins)

        # Find peaks in the histogram
        peaks, _ = find_peaks(counts, prominence=50)  # Adjust 'prominence' as needed

        # Get the start of the second peak
        if len(peaks) > 1:
            second_peak_start_bin = bin_edges[peaks[1] - 1]  # Get the bin edge before the second peak
            print(f"Second peak starts at: {second_peak_start_bin}")

            # Filter the data to exclude values after the second peak start
            filtered_data = iq_data[iq_data < second_peak_start_bin]



        target_bin_values = calculate_spectrogram(filtered_data)

        # plot_histogram_for_targeted(target_bin_values, i, label=f"{i*2} feet")  
        plot_histogram_for_targeted(target_bin_values, i, label=f"{i*2} feet", title=f"Histogram of Power Values at {target_freq / 1e6} MHz", alpha_mean=0.7, alpha_median=0.7, alpha_mode=0.7)
        # plot_spectrogram_for_targeted(target_bin_values, i, label=f"{i*2} feet")          
        
        print("done  ", i)
        
    # Save the variation plot
    variation_output_path = os.path.join(directory_path, "histogram.png")
    plt.savefig(variation_output_path)
    # plt.show() 

    plt.figure(figsize=(20, 12))

    # Loop through all the files and read the IQ data
    for i, cfile in enumerate(cfile_files, start=1):
        file_path = os.path.join(directory_path, cfile)
        iq_data = read_iq_data(file_path)

        target_bin_values = calculate_spectrogram(iq_data)

        # plot_histogram_for_targeted(target_bin_values, i, label=f"{i*2} feet")  
        plot_histogram_for_targeted(target_bin_values, i, label=f"{i*2} feet", title=f"Histogram of Power Values at {target_freq / 1e6} MHz with mean", alpha_mean=0.7, alpha_median=0, alpha_mode=0)
        # plot_spectrogram_for_targeted(target_bin_values, i, label=f"{i*2} feet")          
        
        print("done  ", i)
        
    # Save the variation plot
    variation_output_path = os.path.join(directory_path, "histogram_mean.png")
    plt.savefig(variation_output_path)
    # plt.show() 

    plt.figure(figsize=(20, 12))

    # Loop through all the files and read the IQ data
    for i, cfile in enumerate(cfile_files, start=1):
        file_path = os.path.join(directory_path, cfile)
        iq_data = read_iq_data(file_path)

        target_bin_values = calculate_spectrogram(iq_data)

        # plot_histogram_for_targeted(target_bin_values, i, label=f"{i*2} feet")  
        plot_histogram_for_targeted(target_bin_values, i, label=f"{i*2} feet", title=f"Histogram of Power Values at {target_freq / 1e6} MHz with median", alpha_mean=0, alpha_median=0.7, alpha_mode=0)
        # plot_spectrogram_for_targeted(target_bin_values, i, label=f"{i*2} feet")          
        
        print("done  ", i)
        
    # Save the variation plot
    variation_output_path = os.path.join(directory_path, "histogram_median.png")
    plt.savefig(variation_output_path)
    # plt.show() 


    plt.figure(figsize=(20, 12))

    # Loop through all the files and read the IQ data
    for i, cfile in enumerate(cfile_files, start=1):
        file_path = os.path.join(directory_path, cfile)
        iq_data = read_iq_data(file_path)

        target_bin_values = calculate_spectrogram(iq_data)

        # plot_histogram_for_targeted(target_bin_values, i, label=f"{i*2} feet")  
        plot_histogram_for_targeted(target_bin_values, i, label=f"{i*2} feet", title=f"Histogram of Power Values at {target_freq / 1e6} MHz with mode", alpha_mean=0, alpha_median=0, alpha_mode=0.7)
        # plot_spectrogram_for_targeted(target_bin_values, i, label=f"{i*2} feet")          
        
        print("done  ", i)
        
    # Save the variation plot
    variation_output_path = os.path.join(directory_path, "histogram_mode.png")
    plt.savefig(variation_output_path)
    plt.show() 



def plot_for_eachfile(cfile_files):

    # Loop through all the files and read the IQ data
    for i, cfile in enumerate(cfile_files, start=1):
        file_path = os.path.join(directory_path, cfile)
        iq_data = read_iq_data(file_path)

        initial_target_bin_values = calculate_spectrogram(iq_data)
        no_of_bins = int(np.sqrt(len(initial_target_bin_values)))
        print("bins   ",no_of_bins)

        # Generate histogram data
        counts, bin_edges = np.histogram(iq_data, bins=no_of_bins)

        # Find peaks in the histogram
        peaks, _ = find_peaks(counts, prominence=50)  # Adjust 'prominence' as needed

        # Get the start of the second peak
        if len(peaks) > 1:
            second_peak_start_bin = bin_edges[peaks[1] - 1]  # Get the bin edge before the second peak
            print(f"Second peak starts at: {second_peak_start_bin}")

            # Filter the data to exclude values after the second peak start
            filtered_data = iq_data[iq_data < second_peak_start_bin]



        target_bin_values = calculate_spectrogram(filtered_data)
        print("bins   ",target_bin_values)
        
        plt.figure(figsize=(20, 12))

        # plot_histogram_for_targeted(target_bin_values, i, label=f"{i*2} feet") 
        plot_histogram_for_targeted(target_bin_values, i, label=f"{i*2} feet", title=f"Histogram of Power Values at {target_freq / 1e6} MHz", alpha_mean=0.7, alpha_median=0.7, alpha_mode=0.7) 
        # plot_spectrogram_for_targeted(target_bin_values, i, label=f"{i*2} feet")  

        print("done  ", i)
        
        # Save the variation plot
        output_path = os.path.join(directory_path, f"hist{cfile}.png")
        # output_path = os.path.join(directory_path, f"spec{cfile}.png")

        plt.savefig(output_path)

        
        # plt.show()



cfile_files = [f for f in os.listdir(directory_path) if f.endswith('.cfile')]
print(cfile_files)

sorted_cfile_files = sorted(cfile_files, key=lambda x: int(x.split('_')[1].split('f')[0]))

print(sorted_cfile_files)

# allin1plot(sorted_cfile_files)
plot_for_eachfile(sorted_cfile_files)
# all_stat(sorted_cfile_files)

