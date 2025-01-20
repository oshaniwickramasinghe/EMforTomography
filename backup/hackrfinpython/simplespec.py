#this has a problem with file type
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import mmap
import plotly.graph_objects as go
import plotly.offline as pyo
import plotly.graph_objs as go
from scipy.stats import mode

iq_data_file = "/media/oshani/Shared/UBUNTU/EMforTomography/hackrfinpython/myrec.cfile"
# iq_data_file = "/media/oshani/Shared/UBUNTU/EMforTomography/data-sandali/3tile/3tilewosandali794.cfile"

samples = np.fromfile('myrec.cfile', dtype=np.int8)
samples = samples[::2] + 1j * samples[1::2]
print(len(samples))
print(samples[0:10])
print(np.max(samples))

# Parameters
sampling_frequency = 20000000  # 20 MHz
center_frequency = 794000000   # MHz
# start_time = 2  # seconds
# end_time = 3   # seconds

graph_title="sampling range = 20MHz,  center frequency = 794MHz,  time= from 2s to 3s, state= desktop turned on 2"

def read_iq_data(file_path):
    # Open the file and map it
    print("start reading")
    with open(file_path, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            # Determine total samples
            file_size = mm.size()
            # num_samples = file_size // (2 * np.dtype(np.float32).itemsize)

            # Read all data at once and close mmap safely
            raw_data = np.frombuffer(mm, dtype=np.int8).copy()  # Copying to avoid BufferError
            iq_data = raw_data[0::2] + 1j * raw_data[1::2]  # Create complex IQ data
    print("done reading")
    return iq_data

# Read IQ data
iq_data = read_iq_data(iq_data_file)
print("printing IQ")
print(len(iq_data))
print(iq_data)

# # Calculate start and end sample indices
# start_sample = int(start_time * sampling_frequency)
# end_sample = int(end_time * sampling_frequency)

# # Extract the segment of interest
# segment = iq_data[start_sample:end_sample]


print("start FFT")
# Perform FFT-based spectrogram
frequencies, times, Sxx = spectrogram(
    iq_data, 
    fs=sampling_frequency, 
    window='hann',  # for smoother FFT
    nperseg=1024,   # Number of samples per segment
    noverlap=512,   # Overlap between segments
    nfft=2048,      # FFT points
    scaling='density'
)
print("done FFT")


# Convert power to dB scale
Sxx_dB = 10 * np.log10(Sxx + 1e-12)  # smoothening

# Generate frequency ticks with 0.1 MHz gap
# y_ticks = np.arange(freq_min, freq_max + 0.1e6, 0.1e6)  # Tick positions
# y_tick_labels = [f"{freq / 1e6:.1f}" for freq in y_ticks]  # Format as MHz

times += 0

print("start spectrogram")
# Create a heatmap 
fig = go.Figure(
    data=go.Heatmap(
        z=Sxx_dB, 
        x=times, 
        y=frequencies, 
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
        range=[0, frequencies],  # Limit to masked frequencies
        # tickvals=y_ticks,  # Specify tick positions
        # ticktext=y_tick_labels,  # Custom labels
        showgrid=True  # Enable gridlines on the y-axis
    ),
    template="plotly_dark"
    )
print("done spectrogram")

fig.show()
