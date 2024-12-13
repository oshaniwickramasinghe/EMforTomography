#from chatgpt
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from scipy.stats import mode
import plotly.graph_objects as go
from python_hackrf import pyhackrf 


# Parameters
sampling_frequency = 20e6  # 20 MHz
center_frequency = 794e6   # MHz
freq_min = 791e6
freq_max = 793e6
graph_title = "Real-Time Spectrogram: HackRF"

# Configure HackRF
hackrf = HackRF()

def configure_hackrf():
    hackrf.sample_rate = sampling_frequency
    hackrf.center_freq = center_frequency
    hackrf.lna_gain = 16
    hackrf.vga_gain = 20
    hackrf.amp_enable = False

# Define frequency range of interest
def calculate_frequency_ticks():
    y_ticks = np.arange(freq_min, freq_max + 0.1e6, 0.1e6)
    y_tick_labels = [f"{freq / 1e6:.1f}" for freq in y_ticks]
    return y_ticks, y_tick_labels

y_ticks, y_tick_labels = calculate_frequency_ticks()

# Set up live plot
fig = go.Figure()
heatmap = go.Heatmap(
    z=[],
    x=[],
    y=[],
    colorscale='Jet',
    colorbar=dict(title='Power (dB)')
)
fig.add_trace(heatmap)
fig.update_layout(
    title=graph_title,
    xaxis=dict(title="Time (s)"),
    yaxis=dict(
        title="Frequency (MHz)",
        range=[freq_min, freq_max],
        tickvals=y_ticks,
        ticktext=y_tick_labels
    ),
    template="plotly_dark"
)

def process_iq_data(iq_data):
    # Perform FFT-based spectrogram
    frequencies, times, Sxx = spectrogram(
        iq_data,
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

    # Convert power to dB scale
    Sxx_dB = 10 * np.log10(Sxx_filtered + 1e-12)
    return frequencies_filtered, Sxx_dB

def update_plot(fig, frequencies, Sxx_dB, times):
    heatmap.z = Sxx_dB
    heatmap.x = times
    heatmap.y = frequencies
    fig.update_traces(heatmap)
    fig.show()

def stream_and_plot():
    configure_hackrf()

    def callback(iq_bytes, _):
        # Convert to complex IQ data
        iq_data = np.frombuffer(iq_bytes, dtype=np.int8).astype(np.float32).view(np.complex64)

        # Process the IQ data
        frequencies, Sxx_dB = process_iq_data(iq_data)

        # Update plot
        update_plot(fig, frequencies, Sxx_dB, times=np.arange(len(Sxx_dB[0])))

    # Start streaming data from HackRF
    hackrf.start_rx_mode(callback)

try:
    stream_and_plot()
except KeyboardInterrupt:
    hackrf.stop_rx_mode()
    hackrf.close()
    print("Streaming stopped.")
