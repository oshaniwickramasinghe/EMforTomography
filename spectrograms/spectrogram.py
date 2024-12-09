#this works 
# #change 
# center_frequency = 299e6   # MHz
# start_time = 5  # seconds
# end_time = 7   # seconds
# freq_min = 296e6
# freq_max = 298e6
# title="sampling range = 20MHz,  center frequency = 299MHz,  time= from 5s to 7s",
# target frequency

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import mmap
import plotly.graph_objects as go
import plotly.offline as pyo
import plotly.graph_objs as go
from scipy.stats import mode

iq_data_file = "/media/oshani/Shared/UBUNTU/EMforTomography/data-sandali/3tile/3tilewosandali794.cfile"

# Parameters
sampling_frequency = 20e6  # 20 MHz
center_frequency = 794e6   # MHz
start_time = 2  # seconds
end_time = 3   # seconds

# Define frequency range of interest
freq_min = 791e6
freq_max = 793e6

graph_title="sampling range = 20MHz,  center frequency = 794MHz,  time= from 2s to 3s, state= desktop turned on 2"

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
iq_data = read_iq_data(iq_data_file)

# Calculate start and end sample indices
start_sample = int(start_time * sampling_frequency)
end_sample = int(end_time * sampling_frequency)

# Extract the segment of interest
segment = iq_data[start_sample:end_sample]

print("start FFT")
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
print("done FFT")

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
mode_power = mode(Sxx_dB, keepdims=True)

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

print("start spectrogram")
# Create a heatmap 
fig = go.Figure(
    data=go.Heatmap(
        z=Sxx_dB, 
        x=times, 
        y=frequencies_filtered, 
        # colorscale='Viridis', 
        colorscale='Jet',
        colorbar=dict(title='Power (dB)')
    )
)

# Update layout
fig.update_layout(
    title=graph_title,
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
    ),
    template="plotly_dark"
    )
print("done spectrogram")
# Show the plot
fig.show()


# Frequency of interest
target_frequency = 792e6  # 297.5 MHz

# Find the closest frequency index
freq_index = np.abs(frequencies_filtered - target_frequency).argmin()

# Extract the power values for the selected frequency bin
power_values = Sxx_dB[freq_index, :]

# Compute statistics
min_power = np.min(power_values)
max_power = np.max(power_values)
mean_power = np.mean(power_values)
mode_power = mode(Sxx_dB, keepdims=True)

# Print results
print(f"Frequency Bin: {frequencies_filtered[freq_index] / 1e6:.2f} MHz")
print(f"Minimum Power (dB): {min_power}")
print(f"Maximum Power (dB): {max_power}")
print(f"Mean Power (dB): {mean_power}")
print(f"Mode Power (dB): {mode_power}")




















# # Plot with plotly
# trace = [go.Heatmap(
#     x= times,
#     y= frequencies_filtered,
#     z= Sxx_dB,
#     colorscale='Jet',
#     )]
# layout = go.Layout(
#     title = 'Spectrogram with plotly',
#     yaxis = dict(title = 'Frequency'), # x-axis label
#     xaxis = dict(title = 'Time'), # y-axis label
#     )
# fig = go.Figure(data=trace, layout=layout)
# pyo.iplot(fig, filename='Spectrogram')



# # Plot the spectrogram
# plt.figure(figsize=(12, 8))
# plt.pcolormesh(times, frequencies_filtered, Sxx_dB, shading='gouraud', cmap='viridis')
# plt.colorbar(label='Power (dB)')
# plt.title("Spectrogram (295-300 MHz)")
# plt.xlabel("Time (s)")
# plt.ylabel("Frequency (Hz)")
# plt.show()