#this works, 2 FFTs on same plot
import mmap
import numpy as np
import matplotlib.pyplot as plt

def read_iq_data(file_path):
    # Open the file and map it
    with open(file_path, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            # Determine total samples
            file_size = mm.size()
            num_samples = file_size // (2 * np.dtype(np.float32).itemsize)

            # Read all data at once and close mmap safely
            raw_data = np.frombuffer(mm, dtype=np.float32).copy()  # Copying to avoid BufferError
            iq_data = raw_data[0::2] + 1j * raw_data[1::2]  # Create complex IQ data

    return iq_data

def plot_fft(iq_data1, iq_data2, sampling_rate):
    # Compute FFT for both IQ datasets
    fft1 = np.fft.fftshift(np.fft.fft(iq_data1))
    fft2 = np.fft.fftshift(np.fft.fft(iq_data2))

    # Frequency axis for plotting
    n = len(iq_data1)
    freqs = np.fft.fftshift(np.fft.fftfreq(n, d=1/sampling_rate))

    # Plot FFT magnitude for both datasets
    plt.figure(figsize=(12, 6))
    plt.plot(freqs, np.abs(fft1), color='blue', label='800MHz with stress')
    plt.plot(freqs, np.abs(fft2), color='red', label='800MHz without stress')
    
    # Adding labels and legend
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title("FFT Comparison of 800MHz band with and without stress")
    plt.legend()
    plt.grid(True)
    plt.show()

# File paths for the two data files
file_path1 = "./chunks/01tile800MYESstrs10MB.cfile"
file_path2 = "./chunks/01tile800MNOstrs10MB.cfile"

# Read data from files
IQlist1 = read_iq_data(file_path1)
IQlist2 = read_iq_data(file_path2)

# Sampling rate (set this according to your data's sampling rate)
sampling_rate = 20e6  # Example: 20 MHz

# Plot FFT of both datasets
plot_fft(IQlist1, IQlist2, sampling_rate)
