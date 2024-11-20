#this works create multiple specs and plots in the same window
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import mmap
import plotly.graph_objects as go
from scipy.stats import mode

iq_data_file = "D:/UCSC/year_4/EMdata/obj3/peaks/0297MHz_1.cfile"  

# Parameters
sampling_frequency = 20e6  # 20 MHz
center_frequency = 299e6   # MHz
start_time = 10  # seconds
end_time = 12   # seconds

# Define frequency range of interest
freq_min = 296e6
freq_max = 298e6

def read_iq_data(file_path):
    with open(file_path, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            file_size = mm.size()
            num_samples = file_size // (2 * np.dtype(np.float32).itemsize)
            raw_data = np.frombuffer(mm, dtype=np.float32).copy()
            iq_data = raw_data[0::2] + 1j * raw_data[1::2]  # Create complex IQ data
    return iq_data

# Read IQ data
iq_data = read_iq_data(iq_data_file)

# Initialize combined spectrogram data
combined_sxx_dB = []
combined_times = []

for t in range(start_time, end_time):
    start_sample = int(t * sampling_frequency)
    end_sample = int((t + 1) * sampling_frequency)
    
    segment = iq_data[start_sample:end_sample]
    
    # Perform FFT-based spectrogram
    frequencies, times, Sxx = spectrogram(
        segment, 
        fs=sampling_frequency, 
        window='hann', 
        nperseg=1024, 
        noverlap=512, 
        nfft=2048, 
        scaling='density'
    )
    
    frequencies_shifted = frequencies + (center_frequency - sampling_frequency / 2)
    mask = (frequencies_shifted >= freq_min) & (frequencies_shifted <= freq_max)
    frequencies_filtered = frequencies_shifted[mask]
    Sxx_filtered = Sxx[mask, :]
    
    Sxx_dB = 10 * np.log10(Sxx_filtered + 1e-12)
    
    # Append spectrogram data
    combined_sxx_dB.append(Sxx_dB)
    combined_times.extend(times + t)

# Concatenate the spectrograms into a single matrix
combined_sxx_dB = np.hstack(combined_sxx_dB)

# Generate frequency ticks
y_ticks = np.arange(freq_min, freq_max + 0.1e6, 0.1e6)
y_tick_labels = [f"{freq / 1e6:.1f}" for freq in y_ticks]

# Create heatmap
fig = go.Figure(
    data=go.Heatmap(
        z=combined_sxx_dB, 
        x=combined_times, 
        y=frequencies_filtered, 
        colorscale='Jet',
        colorbar=dict(title='Power (dB)')
    )
)

# Update layout
fig.update_layout(
    title="Spectrogram (296-298 MHz, 10-15 seconds)",
    xaxis=dict(
        title="Time (s)",
        showgrid=True,
        rangeslider=dict(visible=True),
    ),
    yaxis=dict(
        title="Frequency (MHz)",
        range=[freq_min, freq_max],
        tickvals=y_ticks,
        ticktext=y_tick_labels,
        showgrid=True
    )
)

# Show the plot
fig.show()
