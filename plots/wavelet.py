import mmap
import numpy as np
import pywt
import matplotlib.pyplot as plt

def read_iq_data(file_path):
    # Open the file and map it
    with open(file_path, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            # Determine total samples
            file_size = mm.size()
            # print(file_size)
            num_samples = file_size // (2 * np.dtype(np.float32).itemsize)

            # Read all data at once and close mmap safely
            raw_data = np.frombuffer(mm, dtype=np.float32).copy()  # Copying to avoid BufferError
            iq_data = raw_data[0::2] + 1j * raw_data[1::2]  # Create complex IQ data

    return iq_data,num_samples


file_path= "./chunks/01tile800MNOstrs.cfile"
# file_path = "./1tile/1tile800MNOstrs.cfile"

IQlist,total_elements = read_iq_data(file_path)

# Choose a wavelet type (e.g., 'db1' for Daubechies 1, 'haar' for Haar)
wavelet = 'db1'

# Apply the Discrete Wavelet Transform (DWT)
coeffs = pywt.wavedec(IQlist, wavelet)

num_plots_per_figure = 13

# Plot the wavelet coefficients
plt.figure(figsize=(20, 20))
plt.rcParams.update({'font.size': 5})  

for i in range(num_plots_per_figure):
    plt.subplot(num_plots_per_figure, 1, i + 1)
    plt.subplots_adjust(hspace=2, wspace=1)
    plt.plot(np.abs(coeffs[i]))
    plt.title(f'Level {i}')

plt.xlabel('Coefficient Index')
plt.ylabel('Magnitude')
plt.tight_layout()
plt.show()

# Plot next 13 coefficients
plt.figure(figsize=(20, 20))
plt.rcParams.update({'font.size': 5})  

for i in range(num_plots_per_figure):
    plt.subplot(num_plots_per_figure, 1, i + 1)
    plt.subplots_adjust(hspace=2, wspace=1)
    plt.plot(np.abs(coeffs[i + num_plots_per_figure]))
    plt.title(f'Level {i + num_plots_per_figure}')

plt.xlabel('Coefficient Index')
plt.ylabel('Magnitude')
plt.tight_layout()
plt.show()