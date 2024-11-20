#this works gives coulours for above and below threshold values
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import mmap
import plotly.graph_objects as go
import plotly.offline as pyo
import plotly.graph_objs as go
from scipy.stats import mode

iq_data_file = "D:/UCSC/year_4/EMdata/obj3/peaks/0297MHz_1.cfile"  

# Parameters
sampling_frequency = 20e6  # 20 MHz
center_frequency = 299e6   # MHz
start_time = 5  # seconds
end_time = 6   # seconds

# Define frequency range of interest
freq_min = 296e6
freq_max = 298e6

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

# Read IQ data
iq_data = read_iq_data(iq_data_file)

# Calculate start and end sample indices
start_sample = int(start_time * sampling_frequency)
end_sample = int(end_time * sampling_frequency)

# Extract the segment of interest
segment = iq_data[start_sample:end_sample]

# Perform FFT-based spectrogram
frequencies, times, Sxx = spectrogram(
    segment, 
    fs=sampling_frequency, 
    window='hann',  # for smoother FFT
    nperseg=1024,   # Number of samples per segment
    noverlap=512,   # Overlap between segments
    nfft=2048,      # FFT points
    scaling='density'
)

# Shift frequencies to account for center frequency
frequencies_shifted = frequencies + (center_frequency - sampling_frequency / 2)

# Mask frequencies 
mask = (frequencies_shifted >= freq_min) & (frequencies_shifted <= freq_max)
frequencies_filtered = frequencies_shifted[mask]
Sxx_filtered = Sxx[mask, :]

# Convert power to dB scale
Sxx_dB = 10 * np.log10(Sxx_filtered + 1e-12)  # Avoid log(0) by adding a small value

# Compute statistics
min_power = np.min(Sxx_dB)  # Minimum value
max_power = np.max(Sxx_dB)  # Maximum value

# Mode calculation (across flattened array for global mode)
mode_power = mode(Sxx_dB.flatten())[0][0]  # Mode value

# Print results
print(f"Minimum Power (dB): {min_power}")
print(f"Maximum Power (dB): {max_power}")
print(f"Mode Power (dB): {mode_power}")

print("Sxx_dB shape:", Sxx_dB.shape)
print("frequencies_filtered length:", len(frequencies_filtered))
print("times length:", len(times))

# Generate frequency ticks with 0.1 MHz gap
y_ticks = np.arange(freq_min, freq_max + 0.1e6, 0.1e6)  # Tick positions
y_tick_labels = [f"{freq / 1e6:.1f}" for freq in y_ticks]  # Format as MHz

times += start_time


# Calculate the mode of Sxx_dB
mode_power = mode(Sxx_dB.flatten())[0][0]

# Create a mask for values above and below the mode
above_mode = Sxx_dB > mode_power

# Define custom colorscale
colorscale = [
    [0.0, 'blue'],  # Start of the range, for values below the mode
    [0.5, 'white'], # Midpoint (mode)
    [1.0, 'red']    # End of the range, for values above the mode
]

# Create a heatmap 
fig = go.Figure(
    data=go.Heatmap(
        z=Sxx_dB, 
        x=times, 
        y=frequencies_filtered, 
        # colorscale='Viridis', 
        colorscale='Jet',
        colorbar=dict(title='Power (dB)'),
        zmid=mode_power 
    )
)

# Update layout
fig.update_layout(
    title="sampling range = 20MHz,  center frequency = 299MHz,  time= from 5s to 7s",
    xaxis=dict(
        title="Time (s)",
        showgrid=True,
        # rangeslider=dict(visible=True),
    ),
    yaxis=dict(
        title="Frequency (MHz)",
        range=[freq_min, freq_max],  # Limit to masked frequencies
        tickvals=y_ticks,  # Specify tick positions
        ticktext=y_tick_labels,  # Custom labels
        showgrid=True  # Enable gridlines on the y-axis
    ))

# Show the plot
fig.show()

