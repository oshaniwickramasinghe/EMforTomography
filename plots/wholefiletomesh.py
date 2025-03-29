import numpy as np
import mmap
import matplotlib.pyplot as plt

# File path of the IQ data file
file_path = './1tile/1tile800MYESstrs.cfile'

# Define the sample rate of the IQ data (in Hz)
sample_rate = 20e6  # 20 MHz

# Function to read the entire IQ data file using mmap
def read_iq_data(file_path):
    # Open the file and map it
    with open(file_path, "rb") as f:#
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            # Determine total samples
            file_size = mm.size()
            num_samples = file_size // (2 * np.dtype(np.float32).itemsize)

            # Read all data at once and close mmap safely
            raw_data = np.frombuffer(mm, dtype=np.float32).copy()  # Copying to avoid BufferError
            iq_data = raw_data[0::2] + 1j * raw_data[1::2]  # Create complex IQ data

    return iq_data

# Read the entire IQ data from the file
iq_data = read_iq_data(file_path)

# Plot the spectrogram with millisecond-level resolution
plt.figure(figsize=(10, 6))
plt.specgram(iq_data, NFFT=20000, Fs=sample_rate, noverlap=15000, mode='magnitude')
plt.colorbar(label='Intensity [dB]')
plt.title("Time-Frequency Graph")
plt.xlabel("Time [s]")
plt.ylabel("Frequency [Hz]")
plt.show()
