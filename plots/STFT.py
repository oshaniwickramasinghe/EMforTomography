#uselss not much detailed
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
import mmap

# Function to read IQ data from file
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

# File paths for the two data files
file_path1 = "./chunks/01tile800MNOstrs10MB.cfile"
file_path2 = "./chunks/01tile800MYESstrs10MB.cfile"

# Read data
IQlist1 = read_iq_data(file_path1)
IQlist2 = read_iq_data(file_path2)

# Parameters for STFT
fs = 20e6  # Sampling frequency
nperseg = 1024  # Segment length for STFT 

# Perform STFT on the IQ data
f1, t1, Zxx1 = stft(IQlist1, fs=fs, nperseg=nperseg)
f2, t2, Zxx2 = stft(IQlist2, fs=fs, nperseg=nperseg)

# Plot the STFT magnitude in 3D for IQlist1
fig = plt.figure(figsize=(14, 6))
ax = fig.add_subplot(121, projection='3d')
T, F = np.meshgrid(t1, f1)

# 3D surface plot for the magnitude of STFT
ax.plot_surface(T, F, np.abs(Zxx1), cmap='viridis')
ax.set_title('STFT for 800M without stress')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Frequency (Hz)')
ax.set_zlabel('Magnitude')

# 3D surface plot for IQlist2
ax2 = fig.add_subplot(122, projection='3d')
T2, F2 = np.meshgrid(t2, f2)
ax2.plot_surface(T2, F2, np.abs(Zxx2), cmap='plasma')
ax2.set_title('STFT for 800M with stress')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Frequency (Hz)')
ax2.set_zlabel('Magnitude')

plt.tight_layout()
plt.show()
