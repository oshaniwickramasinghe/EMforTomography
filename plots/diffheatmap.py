import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
import mmap


# Parameters
sampling_rate = 20e6  # 20 MHz

# Load the IQ data from files
# Assuming each file is in a format where each row is a complex IQ sample: "I Q"
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

# Calculate the strength (magnitude) over frequencies for each point in time
def calculate_spectrum(data, sampling_rate):
    # Calculate FFT to get frequency components
    spectrum = fft(data)
    # Calculate the magnitude (strength) of each frequency component
    strength = np.abs(spectrum)
    return strength

# Load both IQ data files
iq_data_1 = read_iq_data("./1tile/1tile800MNOstrs.cfile")
iq_data_2 = read_iq_data("./1tile/1tile800MYESstrs.cfile")

# Calculate the strength of each frequency over time
strength_1 = calculate_spectrum(iq_data_1, sampling_rate)
strength_2 = calculate_spectrum(iq_data_2, sampling_rate)

# Calculate the difference in strengths between corresponding frequencies
strength_diff = np.abs(strength_1 - strength_2)

# Plot the heat map of the magnitude of the difference
plt.figure(figsize=(10, 6))
plt.imshow([strength_diff], aspect='auto', cmap='hot', interpolation='nearest')
plt.colorbar(label="Magnitude of Difference")
plt.xlabel("Frequency (Index)")
plt.ylabel("Magnitude Difference")
plt.title("Difference in Strength Between IQ Data Files (Heat Map)")
plt.show()
