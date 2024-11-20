import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import mmap
from scipy.stats import mode
from scipy import stats


iq_data_file1 = "D:/UCSC/year_4/EMforTomography/spectrograms/Cent299MHz_PowOn2.cfile"  
iq_data_file2 = "D:/UCSC/year_4/EMforTomography/spectrograms/Cent299MHz_Satha.cfile"  


# Parameters
sampling_frequency = 20e6  # 20 MHz
center_frequency = 299e6   # MHz
start_time = 2  # seconds
end_time = 3   # seconds

# Define frequency range of interest
freq_min = 296e6
freq_max = 298e6

def read_iq_data(file_path):
    # Open the file and map it
    print("start reading")
    with open(file_path, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            # Determine total samples
            file_size = mm.size()
            num_samples = file_size // (2 * np.dtype(np.float32).itemsize)

            # Read all data at once and close mmap safely
            raw_data = np.frombuffer(mm, dtype=np.float32).copy()  # Copying to avoid BufferError
            iq_data = raw_data[0::2] + 1j * raw_data[1::2]  # Create complex IQ data
    print("done reading")
    return iq_data

# Read IQ data
iq_data1 = read_iq_data(iq_data_file1)
iq_data2 = read_iq_data(iq_data_file2)

# Calculate start and end sample indices
start_sample = int(start_time * sampling_frequency)
end_sample = int(end_time * sampling_frequency)

# Extract the segment of interest
segment1 = iq_data1[start_sample:end_sample]
segment2 = iq_data2[start_sample:end_sample]


print("start FFT")
# Perform FFT-based spectrogram
frequencies, times, Sxx = spectrogram(
    segment1, 
    fs=sampling_frequency, 
    window='hann',  # for smoother FFT
    nperseg=1024,   # Number of samples per segment
    noverlap=512,   # Overlap between segments
    nfft=2048,      # FFT points
    scaling='density'
)
print("done FFT")
# Shift frequencies to account for center frequency
frequencies_shifted = frequencies + (center_frequency - sampling_frequency / 2)

# Mask frequencies 
mask = (frequencies_shifted >= freq_min) & (frequencies_shifted <= freq_max)
frequencies_filtered = frequencies_shifted[mask]
Sxx_filtered = Sxx[mask, :]

# Convert power to dB scale
Sxx_dB = 10 * np.log10(Sxx_filtered + 1e-12)  # Avoid log(0) by adding a small value

# Frequency of interest
target_frequency = 297.5e6  # 297.5 MHz

# Find the closest frequency index
freq_index = np.abs(frequencies_filtered - target_frequency).argmin()

# Extract the power values for the selected frequency bin
sample_A = Sxx_dB[freq_index, :]


print("start FFT")
# Perform FFT-based spectrogram
frequencies, times, Sxx = spectrogram(
    segment2, 
    fs=sampling_frequency, 
    window='hann',  # for smoother FFT
    nperseg=1024,   # Number of samples per segment
    noverlap=512,   # Overlap between segments
    nfft=2048,      # FFT points
    scaling='density'
)
print("done FFT")

# Shift frequencies to account for center frequency
frequencies_shifted = frequencies + (center_frequency - sampling_frequency / 2)

# Mask frequencies 
mask = (frequencies_shifted >= freq_min) & (frequencies_shifted <= freq_max)
frequencies_filtered = frequencies_shifted[mask]
Sxx_filtered = Sxx[mask, :]

# Convert power to dB scale
Sxx_dB = 10 * np.log10(Sxx_filtered + 1e-12)  # Avoid log(0) by adding a small value

# Frequency of interest
target_frequency = 297.5e6  # 297.5 MHz

# Find the closest frequency index
freq_index = np.abs(frequencies_filtered - target_frequency).argmin()

# Extract the power values for the selected frequency bin
sample_B = Sxx_dB[freq_index, :]

# Perform independent sample t-test
t_statistic, p_value = stats.ttest_ind(sample_A, sample_B)

# Set the significance level (alpha)
alpha = 0.05

# Compute the degrees of freedom (df) (n_A-1)+(n_b-1)
df = len(sample_A)+len(sample_B)-2

# Calculate the critical t-value
# ppf is used to find the critical t-value for a two-tailed test
critical_t = stats.t.ppf(1 - alpha/2, df)


# Print the results
print("T-value:", t_statistic)
print("P-Value:", p_value)
print("Critical t-value:", critical_t)

# Decision
print('With T-value')
if np.abs(t_statistic) >critical_t:
    print('There is significant difference between two groups')
else:
    print('No significant difference found between two groups')

print('With P-value')
if p_value >alpha:
    print('No evidence to reject the null hypothesis')
else:
    print('Evidence found to reject the null hypothesis')
