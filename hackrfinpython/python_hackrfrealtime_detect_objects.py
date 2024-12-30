#this works
#do as in https://pysdr.org/content/hackrf.html#tx-and-rx-gain
# additionally had to install       /hackrf/host/build$ sudo apt install cmake libusb-1.0-0-dev libfftw3-dev
# rm -rf /home/oshani/hackrf/host/build/*     then cmake ..
#change based on the threshold value

from python_hackrf import pyhackrf  # type: ignore
import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.signal import spectrogram


recording_time = 0.1  # seconds
center_freq = 794e6  # Hz
sample_rate = 20e6
baseband_filter = 20e6
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

def rx_callback(device, buffer, buffer_length, valid_length):  # this callback function always needs to have these four args
    global samples, last_idx

    accepted = valid_length // 2
    accepted_samples = buffer[:valid_length].astype(np.int8)  # -128 to 127
    accepted_samples = accepted_samples[0::2] + 1j * accepted_samples[1::2]  # Convert to complex type (de-interleave the IQ)
    accepted_samples /= 128  # -1 to +1
    samples[last_idx: last_idx + accepted] = accepted_samples

    last_idx += accepted

    return 0


# def plot_spectrogram(samples):
#     fft_size = 1024
#     num_rows = len(samples) // fft_size
#     spectrogram = np.zeros((num_rows, fft_size))

#     # Perform FFT and calculate the spectrogram
#     for i in range(num_rows):
#         spectrogram[i, :] = 10 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(samples[i * fft_size:(i+1) * fft_size]))) ** 2)

#     # Calculate the frequency resolution
#     freq_resolution = sample_rate / fft_size  # Hz per bin
#     # Calculate the index for 794 MHz (which is the desired frequency)
#     target_freq = 794e6  # 794 MHz
#     target_bin = int((target_freq - center_freq) / freq_resolution + fft_size // 2)  # Convert to bin index

#     # Extract the power values for the 794 MHz bin
#     target_bin_values = spectrogram[:, target_bin]

#     # Plot the signal strength at 794 MHz
#     plt.figure(1)
#     plt.clf()
#     plt.plot(target_bin_values, linewidth=0.5)    
#     plt.xlabel("Time [s]")
#     plt.ylabel("Signal Strength [dB]")
#     plt.title(f"Signal Strength at {target_freq / 1e6} MHz")
#     plt.grid(True)    
#     plt.pause(0.01)  # Pause to allow the plot to update

sampling_frequency = 20e6  
center_frequency = 794e6  
target_freq = 792e6

def compute_spectrogram(iq_segment):
    """
    Computes the spectrogram and filters the target frequency.
    """
    print("Computing spectrogram...")
    frequencies, times, Sxx = spectrogram(
        iq_segment,
        fs=sampling_frequency,
        window='hann',
        nperseg=1024,
        noverlap=512,
        nfft=2048,
        scaling='density'
    )

    frequencies_shifted = frequencies + (center_frequency - sampling_frequency / 2)

    target_index = np.abs(frequencies_shifted - target_freq).argmin()
    print("Spectrogram computation complete.")

    return times, 10 * np.log10(Sxx[target_index, :])  # Convert power to dB
    # return times, 10 * np.log10(Sxx[target_index, :] + 1e-12)  # Convert power to dB

def plot_spectrogram_for_targeted(target_bin_values):
   
    plt.figure(1)
    plt.clf()
    plt.plot(target_bin_values, linewidth=0.5)    
    plt.xlabel("Time [s]")
    plt.ylabel("Signal Strength [dB]")
    plt.title(f"Signal Strength at {target_freq / 1e6} MHz")
    plt.grid(True)    
    plt.pause(0.01)  # Pause to allow the plot to update



sdr.set_rx_callback(rx_callback)
sdr.pyhackrf_start_rx()
print('is_streaming', sdr.pyhackrf_is_streaming())

try:
    while True:  # Continuous loop
        # global last_idx
        samples[:] = 0  # Clear the samples buffer
        last_idx = 0  # Reset the index

        time.sleep(recording_time)  # Record for the specified time
        times, bin_values = compute_spectrogram(samples)
        plot_spectrogram_for_targeted(bin_values)  # Plot the spectrogram

except KeyboardInterrupt:
    print("Stopping data collection...")

sdr.pyhackrf_stop_rx()
sdr.pyhackrf_close()
