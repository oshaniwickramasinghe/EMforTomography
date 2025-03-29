import mmap
import numpy as np
import pywt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

def compute_wavelet_transform(iq_data):
    # Use CWT with the 'cmor' (complex Morlet) wavelet, suitable for complex-valued data
    scales = np.arange(1, 100)  # Define a range of scales for the wavelet transform
    coefficients, frequencies = pywt.cwt(iq_data, scales, 'cmor')  # Compute the CWT
    return coefficients, frequencies

def plot_3d_wavelet(coefficients, frequencies, title):
    # Create a 3D plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # Prepare data for 3D plotting
    t = np.arange(coefficients.shape[1])  # Time axis
    T, F = np.meshgrid(t, frequencies)
    Z = np.abs(coefficients)  # Magnitude of wavelet coefficients for visualization

    # Plot the 3D surface
    surf = ax.plot_surface(T, F, Z, cmap='viridis', edgecolor='none')
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Frequency")
    ax.set_zlabel("Magnitude")
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Magnitude')
    plt.show()

# File paths for the two data files
file_path1 = "./chunks/01tile800MNOstrs10MB.cfile"
# file_path2 = "./chunks/01tile800MNOstrs.cfile"

# Read data and compute wavelet coefficients for both files
IQlist1 = read_iq_data(file_path1)
# IQlist2 = read_iq_data(file_path2)

# Compute wavelet transformations
coefficients1, frequencies1 = compute_wavelet_transform(IQlist1)
# coefficients2, frequencies2 = compute_wavelet_transform(IQlist2)

# Plot in 3D
plot_3d_wavelet(coefficients1, frequencies1, "Wavelet Transform of File 1")
# plot_3d_wavelet(coefficients2, frequencies2, "Wavelet Transform of File 2")
