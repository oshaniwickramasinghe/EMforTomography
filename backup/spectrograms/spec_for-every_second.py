import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import mmap
import plotly.graph_objects as go
from scipy.stats import mode

# file_name = "filtered_794_1t_null.cfile"
# iq_data_file = f"/media/oshani/Shared/UBUNTU/EMforTomography/794_noob_filtered/{file_name}"

file_name = "794_1t_null.cfile"
iq_data_file = f"/media/oshani/Shared/UBUNTU/EMforTomography/794/no_object/{file_name}"


# Parameters
sampling_frequency = 20e6  # 20 MHz
center_frequency = 794e6   # MHz
freq_min = 791e6
freq_max = 793e6

# Function to read IQ data
def read_iq_data(file_path):
    print("Start reading data")
    with open(file_path, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            file_size = mm.size()
            num_samples = file_size // (2 * np.dtype(np.float32).itemsize)
            raw_data = np.frombuffer(mm, dtype=np.float32).copy()
            iq_data = raw_data[0::2] + 1j * raw_data[1::2]
    print("Finished reading data")
    return iq_data

# Function to generate spectrogram
def generate_spectrogram(iq_segment, start_time, end_time, interval_index):
    print(f"Generating spectrogram for interval {start_time}s to {end_time}s")
    frequencies, times, Sxx = spectrogram(
        iq_segment,
        fs=sampling_frequency,
        window='hann',
        nperseg=1024,
        noverlap=512,
        nfft=2048,
        scaling='density'
    )
    
    # Shift frequencies to account for center frequency
    frequencies_shifted = frequencies + (center_frequency - sampling_frequency / 2)
    
    # Mask frequencies
    mask = (frequencies_shifted >= freq_min) & (frequencies_shifted <= freq_max)
    frequencies_filtered = frequencies_shifted[mask]
    Sxx_filtered = Sxx[mask, :]
    
    # Convert power to dB
    Sxx_dB = 10 * np.log10(Sxx_filtered + 1e-12)
    # Sxx_dB = 10 * np.log10(Sxx_filtered)

    
    # Generate spectrogram using Plotly
    times += start_time
    fig = go.Figure(
        data=go.Heatmap(
            z=Sxx_dB,
            x=times,
            y=frequencies_filtered,
            colorscale='Jet',
            # zmin=-120,  # Set the minimum value for the color scale
            # zmax=-95,     # Set the maximum value for the color scale
            colorbar=dict(title='Power (dB)')
        )
    )
    fig.update_layout(
        title=f"Spectrogram (Interval {start_time}s to {end_time}s)",
        xaxis=dict(title="Time (s)"),
        yaxis=dict(
            title="Frequency (MHz)",
            tickvals=np.arange(freq_min, freq_max + 0.1e6, 0.1e6),
            ticktext=[f"{freq / 1e6:.1f}" for freq in np.arange(freq_min, freq_max + 0.1e6, 0.1e6)],
            range=[freq_min, freq_max]
        ),
        template="plotly_dark"
    )
    fig.show()
    print(f"Spectrogram for interval {start_time}s to {end_time}s displayed of {file_name}")


# Main processing loop
iq_data = read_iq_data(iq_data_file)
total_samples = len(iq_data)
interval_duration = 1  # second
samples_per_interval = int(sampling_frequency * interval_duration)

interval_index = 0
while True:
    start_sample = interval_index * samples_per_interval
    end_sample = start_sample + samples_per_interval
    
    if end_sample > total_samples:
        print(f"No more data to process at interval {interval_index}. Exiting loop.")
        break
    
    # Extract the segment
    iq_segment = iq_data[start_sample:end_sample]
    
    # Generate spectrogram
    generate_spectrogram(
        iq_segment,
        start_time=interval_index * interval_duration,
        end_time=(interval_index + 1) * interval_duration,
        interval_index=interval_index
    )
    
    interval_index += 1
