from python_hackrf import pyhackrf  # type: ignore
import numpy as np
import time
import plotly.graph_objects as go
import plotly.offline as pyo

recording_time = 1  # seconds
center_freq = 794e6  # Hz
sample_rate = 10e6
baseband_filter = 7.5e6
lna_gain = 30  # 0 to 40 dB in 8 dB steps
vga_gain = 50  # 0 to 62 dB in 2 dB steps

pyhackrf.pyhackrf_init()
sdr = pyhackrf.pyhackrf_open()

allowed_baseband_filter = pyhackrf.pyhackrf_compute_baseband_filter_bw_round_down_lt(baseband_filter)  # calculate the supported bandwidth relative to the desired one

sdr.pyhackrf_set_sample_rate(sample_rate)
sdr.pyhackrf_set_baseband_filter_bandwidth(allowed_baseband_filter)
sdr.pyhackrf_set_antenna_enable(False)  # It seems this setting enables or disables power supply to the antenna port. False by default. the firmware auto-disables this after returning to IDLE mode

sdr.pyhackrf_set_freq(center_freq)
sdr.pyhackrf_set_amp_enable(False)  # False by default
sdr.pyhackrf_set_lna_gain(lna_gain)  # LNA gain - 0 to 40 dB in 8 dB steps
sdr.pyhackrf_set_vga_gain(vga_gain)  # VGA gain - 0 to 62 dB in 2 dB steps

print(f'center_freq: {center_freq} sample_rate: {sample_rate} baseband_filter: {allowed_baseband_filter}')

num_samples = int(recording_time * sample_rate)
samples = np.zeros(num_samples, dtype=np.complex64)
last_idx = 0

# Create the figures for the spectrogram and amplitude plots
fft_size = 2048
time_axis = np.linspace(0, len(samples) / sample_rate, num_samples // fft_size)
frequency_axis = np.linspace(center_freq - sample_rate / 2, center_freq + sample_rate / 2, fft_size) / 1e6

# Initialize figures for spectrogram and amplitude
fig_spectrogram = go.Figure()
fig_amplitude = go.Figure()

def rx_callback(device, buffer, buffer_length, valid_length):  # this callback function always needs to have these four args
    global samples, last_idx

    accepted = valid_length // 2
    accepted_samples = buffer[:valid_length].astype(np.int8)  # -128 to 127
    accepted_samples = accepted_samples[0::2] + 1j * accepted_samples[1::2]  # Convert to complex type (de-interleave the IQ)
    accepted_samples /= 128  # -1 to +1
    samples[last_idx: last_idx + accepted] = accepted_samples

    last_idx += accepted

    return 0

def update_spectrogram(samples):
    spectrogram = np.zeros((len(samples) // fft_size, fft_size))

    for i in range(len(samples) // fft_size):
        spectrogram[i, :] = 10 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(samples[i * fft_size:(i+1) * fft_size]))) ** 2)

    fig_spectrogram.data = []  # Clear the previous data
    fig_spectrogram.add_trace(go.Heatmap(
        z=spectrogram,
        x=frequency_axis,
        y=time_axis,
        colorscale='Viridis',
        colorbar=dict(title='dB')
    ))

    fig_spectrogram.update_layout(
        title='Spectrogram',
        xaxis_title='Frequency [MHz]',
        yaxis_title='Time [s]',
        showlegend=False
    )

    fig_spectrogram.update()

def update_amplitude_plot(samples):
    fig_amplitude.data = []  # Clear the previous data
    fig_amplitude.add_trace(go.Scatter(
        x=np.arange(10000),
        y=np.real(samples[:10000]),
        mode='lines',
        name='Real'
    ))

    fig_amplitude.add_trace(go.Scatter(
        x=np.arange(10000),
        y=np.imag(samples[:10000]),
        mode='lines',
        name='Imaginary'
    ))

    fig_amplitude.update_layout(
        title="Real and Imaginary Parts of the Samples",
        xaxis_title="Samples",
        yaxis_title="Amplitude",
        legend_title="Component"
    )

    fig_amplitude.update()

sdr.set_rx_callback(rx_callback)
sdr.pyhackrf_start_rx()
print('is_streaming', sdr.pyhackrf_is_streaming())

try:
    while True:  # Continuous loop
        samples[:] = 0  # Clear the samples buffer
        last_idx = 0  # Reset the index

        time.sleep(recording_time)  # Record for the specified time
        update_spectrogram(samples)  # Update the spectrogram
        update_amplitude_plot(samples)  # Update the amplitude plot

        # Display the updated plots
        pyo.iplot(fig_spectrogram)
        pyo.iplot(fig_amplitude)

except KeyboardInterrupt:
    print("Stopping data collection...")

sdr.pyhackrf_stop_rx()
sdr.pyhackrf_close()
