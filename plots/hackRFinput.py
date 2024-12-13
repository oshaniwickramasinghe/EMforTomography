import numpy as np
import hackrf
import time
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Capture IQ data from HackRF
def capture_from_hackrf(duration_sec=1, frequency=28e6, sample_rate=10e6):
    device = hackrf.HackRF()
    device.setup()

    device.set_freq(frequency)  # Set frequency (e.g., 28 MHz)
    device.set_sample_rate(sample_rate)  # Set sample rate (e.g., 10 MSPS)

    # Start receiving data from HackRF
    device.start_rx()
    
    # Capture data for the given duration
    iq_data = device.read_samples(duration_sec * int(sample_rate))  # Number of samples = duration * sample rate
    device.stop_rx()
    
    # Return the captured IQ data
    return iq_data
def save_data_to_file(iq_data, file_path):
    with open(file_path, 'ab') as f:
        iq_data.tofile(f)  # Append IQ data to file


# Convert HackRF samples into complex IQ data
def process_hackrf_data(raw_data):
    iq_data = raw_data[::2] + 1j * raw_data[1::2]  # Separate In-phase (I) and Quadrature (Q)
    return iq_data

# Plot the data as it updates
def update_plot(data, fig):
    real_data = np.abs(data)  # Magnitude of IQ data
    fig.add_trace(go.Scatter(
        x=np.arange(len(real_data)),
        y=real_data,
        mode='lines',
        line=dict(color='blue', width=1)
    ))
    fig.update_layout(
        title="Live HackRF IQ Data",
        xaxis=dict(title="Sample Index"),
        yaxis=dict(title="Amplitude"),
    )

# Initialize the plot
fig = make_subplots(rows=1, cols=1)
fig.update_layout(
    title="Live HackRF IQ Data",
    xaxis=dict(title="Sample Index"),
    yaxis=dict(title="Amplitude")
)

# Loop to capture data every second and update the plot
while True:
    # Capture IQ data from HackRF
    raw_data = capture_from_hackrf(duration_sec=1)

    # Process the raw data into complex IQ format
    iq_data = process_hackrf_data(raw_data)
    # Example usage inside the main loop
    save_data_to_file(iq_data, 'hackrf_data.cfile')

    # Update the plot with the captured data
    update_plot(iq_data, fig)
    
    # Show the plot
    fig.show()

    # Pause for a second before capturing again
    time.sleep(1)
