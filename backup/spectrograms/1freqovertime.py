import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import mmap

# File paths
iq_data_file1 = "/media/oshani/Shared/UBUNTU/EMforTomography/893/dasun/893_4t.cfile"
iq_data_file2 = "/media/oshani/Shared/UBUNTU/EMforTomography/893/dasun/893_4t.cfile"

# Parameters
sampling_frequency = 20e6  # Hz
center_frequency = 893e6  # Hz
freq_target = 800e6  # Hz (Target frequency to analyze)

target_freq = 890e6

start_time1 = 0.4
end_time1 = 1.4
start_time2 = 2.8
end_time2 = 3.7


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


# def compute_spectrogram(iq_segment, sampling_frequency, center_frequency):
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
#     target_index = np.abs(frequencies_shifted - freq_target).argmin()
#     print("Spectrogram computation complete.")
#     return times, 10 * np.log10(Sxx[target_index, :] + 1e-12)  # Convert power to dB

def plot_spectrogram(samples, label):
    fft_size = 1024
    num_rows = len(samples) // fft_size
    spectrogram = np.zeros((num_rows, fft_size))

    # Perform FFT and calculate the spectrogram
    for i in range(num_rows):
        spectrogram[i, :] = 10 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(samples[i * fft_size:(i+1) * fft_size]))) ** 2)

    # Calculate the frequency resolution
    freq_resolution = sampling_frequency / fft_size  # Hz per bin
    # Calculate the index for 794 MHz (which is the desired frequency)
    # target_freq = 794e6  # 794 MHz
    target_bin = int((target_freq - center_frequency) / freq_resolution + fft_size // 2)  # Convert to bin index

    # Extract the power values for the 794 MHz bin
    target_bin_values = spectrogram[:, target_bin]

    # Plot the signal strength at 794 MHz
    plt.plot(target_bin_values, linewidth=0.5, label=label)    
    plt.xlabel("Time [s]")
    plt.ylabel("Signal Strength [dB]")
    plt.title(f"Signal Strength at {target_freq / 1e6} MHz")
    plt.grid(True)
    plt.legend()

   
    
# Main Execution
print("Starting analysis...")

# Read IQ data
iq_data1 = read_iq_data(iq_data_file1)
iq_data2 = read_iq_data(iq_data_file2)

# Extract segments
segment1 = extract_segment(iq_data1, start_time1, end_time1, sampling_frequency)
segment2 = extract_segment(iq_data2, start_time2, end_time2, sampling_frequency)

# Plot the variation of signal strength over time
plt.figure()
plot_spectrogram(iq_data1, label="with")
plot_spectrogram(iq_data2, label="without")
plt.show()

print("Analysis complete.")
