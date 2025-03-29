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
            num_samples = file_size // (2 * np.dtype(np.float32).itemsize)

            # Read all data at once and close mmap safely
            raw_data = np.frombuffer(mm, dtype=np.float32).copy()  # Copying to avoid BufferError
            iq_data = raw_data[0::2] + 1j * raw_data[1::2]  # Create complex IQ data

    return iq_data

# File paths for the two data files
file_path1 = "./1tile/1tile800MYESstrs.cfile"
file_path2 = "./1tile/1tile800MNOstrs.cfile"

# Read data and compute wavelet coefficients for both files
IQlist1 = read_iq_data(file_path1)
IQlist2 = read_iq_data(file_path2)

# Choose a wavelet type
wavelet = 'db3'

# Apply the Discrete Wavelet Transform (DWT)
coeffs1 = pywt.wavedec(IQlist1, wavelet)
coeffs2 = pywt.wavedec(IQlist2, wavelet)

# Number of subplots per figure
num_plots_per_figure = 13

# first 13 levels
plt.figure(figsize=(21, 14))
plt.rcParams.update({'font.size': 5})  

for i in range(num_plots_per_figure):
    plt.subplot(num_plots_per_figure, 1, i + 1)
    plt.subplots_adjust(hspace=1.8, wspace=0.8)
    plt.plot(np.abs(coeffs1[i]), color='blue', label='With Stress')
    plt.plot(np.abs(coeffs2[i]), color='red', label='Without Stress')
    plt.title(f'Level {i}')
    if i == 0:  # Add legend only to the first subplot
        plt.legend()

plt.rcParams.update({'font.size': 8}) 
# plt.title("db1 Wavelet transformation for 800MHz with and without stress (1)") 
plt.xlabel('Coefficient Index')
plt.ylabel('db3 Wavelet transformation for 800MHz with and without stress (1) \n Magnitude')
plt.tight_layout()
plt.show()

# levels 13-25 
plt.figure(figsize=(21, 14))
plt.rcParams.update({'font.size': 5})  

for i in range(num_plots_per_figure):
    level_index = i + num_plots_per_figure
    if level_index < len(coeffs1) and level_index < len(coeffs2):  # Check if the level exists in both
        plt.subplot(num_plots_per_figure, 1, i + 1)
        plt.subplots_adjust(hspace=2, wspace=1)
        plt.plot(np.abs(coeffs1[level_index]), color='blue', label='With Stress')
        plt.plot(np.abs(coeffs2[level_index]), color='red', label='Without Stress')
        plt.title(f'Level {level_index}')
        if i == 0:  # Add legend only to the first subplot
            plt.legend()

plt.rcParams.update({'font.size': 8}) 
# plt.title("db1 Wavelet transformation for 800MHz with and without stress (2)") 
plt.xlabel('Coefficient Index')
plt.ylabel('db3 Wavelet transformation for 800MHz with and without stress (2) \n Magnitude')
plt.tight_layout()
plt.show()
