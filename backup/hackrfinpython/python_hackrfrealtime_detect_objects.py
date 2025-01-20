# from python_hackrf import pyhackrf  # type: ignore
# import matplotlib.pyplot as plt
# import numpy as np
# import time
# from scipy.signal import spectrogram

# recording_time = 0.5  # seconds
# center_freq = 794e6  # Hz
# sample_rate = 20e6
# baseband_filter = 20e6
# lna_gain = 30  # 0 to 40 dB in 8 dB steps
# vga_gain = 50  # 0 to 62 dB in 2 dB steps

# pyhackrf.pyhackrf_init()
# sdr = pyhackrf.pyhackrf_open()

# allowed_baseband_filter = pyhackrf.pyhackrf_compute_baseband_filter_bw_round_down_lt(baseband_filter)

# sdr.pyhackrf_set_sample_rate(sample_rate)
# sdr.pyhackrf_set_baseband_filter_bandwidth(allowed_baseband_filter)
# sdr.pyhackrf_set_antenna_enable(False)
# sdr.pyhackrf_set_freq(center_freq)
# sdr.pyhackrf_set_amp_enable(False)
# sdr.pyhackrf_set_lna_gain(lna_gain)
# sdr.pyhackrf_set_vga_gain(vga_gain)

# print(f'center_freq: {center_freq} sample_rate: {sample_rate} baseband_filter: {allowed_baseband_filter}')

# num_samples = int(recording_time * sample_rate)
# samples = np.zeros(num_samples, dtype=np.complex64)
# last_idx = 0

# collected_data = []
# signal_range = (0, 0)  # Placeholder for signal range

# def rx_callback(device, buffer, buffer_length, valid_length):
#     global samples, last_idx

#     accepted = valid_length // 2
#     accepted_samples = buffer[:valid_length].astype(np.int8)
#     accepted_samples = accepted_samples[0::2] + 1j * accepted_samples[1::2]
#     accepted_samples /= 128
#     samples[last_idx: last_idx + accepted] = accepted_samples

#     last_idx += accepted

#     return 0

# def compute_spectrogram(iq_segment):
#     frequencies, times, Sxx = spectrogram(
#         iq_segment,
#         fs=sample_rate,
#         window='hann',
#         nperseg=1024,
#         noverlap=512,
#         nfft=2048,
#         scaling='density'
#     )
#     frequencies_shifted = frequencies + (center_freq - sample_rate / 2)
#     target_index = np.abs(frequencies_shifted - target_freq).argmin()
#     return times, 10 * np.log10(Sxx[target_index, :]), frequencies_shifted, Sxx

# # def calculate_signal_range(signal_values):
# #     return np.max(signal_values), np.mean(signal_values)

# def calculate_signal_range(signal_values):
#     # Remove infinite values for calculation
#     finite_values = signal_values[np.isfinite(signal_values)]
    
#     if len(finite_values) > 0:
#         max_value = np.max(finite_values)
#         min_value = np.min(finite_values)
#         median_value = np.median(finite_values)
#     else:
#         max_value = min_value = median_value = -np.inf  # Handle case where all values are infinite

#     return max_value, min_value, median_value



# def plot_realtime(target_bin_values, max_value, min_value, median_value):
#     plt.figure(1)
#     plt.clf()

#     # Calculate the median of target_bin_values
#     median_of_values = np.median(target_bin_values)

#     # Plot signal within the range of max and min values
#     target_bin_values = np.clip(target_bin_values, min_value, max_value)

#     below_range = target_bin_values < min_value
#     # plt.plot(target_bin_values, 'b', linewidth=0.5)
#     # plt.plot(np.where(below_range)[0], target_bin_values[below_range], 'r.', label='Below Signal Range')

#     # Plot the median from the calculate_signal_range function as a static line
#     plt.axhline(y=median_value, color='g', linestyle='--', label=f'Calibrated Median : {median_value:.2f} dB')
    
#     # Plot the calculated median of target_bin_values as a static line in a different color
#     plt.axhline(y=median_of_values, color='m', linestyle='-.', label=f'Realtime Median : {median_of_values:.2f} dB')

#     plt.text(x=0, y=median_value, s=f'{median_value:.2f} dB', color='g', fontsize=10, ha='left', va='top', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

#     plt.text(x=0, y=median_of_values, s=f'{median_of_values:.2f} dB', color='m', fontsize=10, ha='left', va='bottom', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))



#     plt.xlabel("Time [s]")
#     plt.ylabel("Signal Strength [dB]")
#     plt.title(f"Signal Strength at {target_freq / 1e6} MHz")
#     plt.legend()
#     plt.grid(True)
#     plt.pause(0.01)




# target_freq = 792e6

# sdr.set_rx_callback(rx_callback)
# sdr.pyhackrf_start_rx()

# # Data collection for 5 seconds
# try:
#     print("Collecting initial data...")
#     for _ in range(int(5 / recording_time)):
#         samples[:] = 0
#         last_idx = 0
#         time.sleep(recording_time)
#         collected_data.append(samples.copy())
#     collected_data = np.concatenate(collected_data)
#     np.save("collected_data.npy", collected_data)

#     # Perform spectrogram analysis
#     print("Analyzing collected data...")
#     _, initial_bin_values, _, _ = compute_spectrogram(collected_data)
#     max_value, min_value, median_value = calculate_signal_range(initial_bin_values)
#     print(f"Signal range for 792 MHz: Max: {max_value}, Min: {min_value}, Median: {median_value}")


#     print("Starting real-time processing...")
#     while True:
#         samples[:] = 0
#         last_idx = 0
#         time.sleep(recording_time)
#         times, bin_values, _, _ = compute_spectrogram(samples)
#         plot_realtime(bin_values, max_value, min_value, median_value)


# except KeyboardInterrupt:
#     print("Stopping data collection...")

# sdr.pyhackrf_stop_rx()
# sdr.pyhackrf_close()






from python_hackrf import pyhackrf  # type: ignore
import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.signal import spectrogram
from collections import deque

recording_time = 1  # seconds
calibration_time = 30  # seconds
center_freq = 794e6  # Hz
sample_rate = 20e6
baseband_filter = 20e6
lna_gain = 30  # 0 to 40 dB in 8 dB steps
vga_gain = 50  # 0 to 62 dB in 2 dB steps

pyhackrf.pyhackrf_init()
sdr = pyhackrf.pyhackrf_open()

allowed_baseband_filter = pyhackrf.pyhackrf_compute_baseband_filter_bw_round_down_lt(baseband_filter)

sdr.pyhackrf_set_sample_rate(sample_rate)
sdr.pyhackrf_set_baseband_filter_bandwidth(allowed_baseband_filter)
sdr.pyhackrf_set_antenna_enable(False)
sdr.pyhackrf_set_freq(center_freq)
sdr.pyhackrf_set_amp_enable(False)
sdr.pyhackrf_set_lna_gain(lna_gain)
sdr.pyhackrf_set_vga_gain(vga_gain)

print(f'center_freq: {center_freq} sample_rate: {sample_rate} baseband_filter: {allowed_baseband_filter}')

num_samples = int(recording_time * sample_rate)
samples = np.zeros(num_samples, dtype=np.complex64)
last_idx = 0

target_freq = 792e6

calibration_medians = []  # Store medians during calibration
def rx_callback(device, buffer, buffer_length, valid_length):
    global samples, last_idx

    accepted = valid_length // 2
    accepted_samples = buffer[:valid_length].astype(np.int8)
    accepted_samples = accepted_samples[0::2] + 1j * accepted_samples[1::2]
    accepted_samples /= 128
    samples[last_idx: last_idx + accepted] = accepted_samples

    last_idx += accepted

    return 0

def compute_spectrogram(iq_segment):
    frequencies, times, Sxx = spectrogram(
        iq_segment,
        fs=sample_rate,
        window='hann',
        nperseg=1024,
        noverlap=512,
        nfft=2048,
        scaling='density'
    )
    frequencies_shifted = frequencies + (center_freq - sample_rate / 2)
    target_index = np.abs(frequencies_shifted - target_freq).argmin()
    return times, 10 * np.log10(Sxx[target_index, :]), frequencies_shifted, Sxx

def calculate_signal_range(signal_values):
    finite_values = signal_values[np.isfinite(signal_values)]
    if len(finite_values) > 0:
        max_value = np.max(finite_values)
        min_value = np.min(finite_values)
        median_value = np.median(finite_values)
    else:
        max_value = min_value = median_value = -np.inf
    return max_value, min_value, median_value

def plot_realtime(times_list, medians, max_value, min_value):
    plt.figure(1)
    plt.clf()

    # Plot median values over time
    print(medians)
    print(times_list)
    plt.plot(times_list, medians, 'b-', label='Realtime Median [dB]')

    # Mark calibration range (min and max)
    plt.axhline(y=max_value, color='r', linestyle='--', label=f'Calibrated Max: {max_value:.2f} dB')
    plt.axhline(y=min_value, color='g', linestyle='--', label=f'Calibrated Min: {min_value:.2f} dB')

    plt.xlim(min(times_list), max(times_list)) 
    plt.ylim(min_value - 5, max_value + 5)

    plt.xlabel("Time [s]")
    plt.ylabel("Signal Strength [dB]")
    plt.title(f"Signal Strength at {target_freq / 1e6} MHz")
    plt.legend()
    plt.grid(True)
    plt.pause(0.01)

sdr.set_rx_callback(rx_callback)
sdr.pyhackrf_start_rx()



try:
    print("Calibrating for 30 seconds...")
    start_time = time.time()

    while time.time() - start_time < calibration_time:
        samples[:] = 0
        last_idx = 0
        time.sleep(recording_time)

        _, bin_values, _, _ = compute_spectrogram(samples)
        _, _, median_value = calculate_signal_range(bin_values)
        calibration_medians.append(median_value)

    # Compute calibration range
    max_value = np.max(calibration_medians)
    min_value = np.min(calibration_medians)
    print(f"Calibration complete. Max: {max_value:.2f} dB, Min: {min_value:.2f} dB")

    print("Starting real-time processing...")
    size = 30
    medians = []
    times_list = []
  
    current_time = 0

    while True:
        samples[:] = 0
        last_idx = 0
        time.sleep(recording_time)

        # Keep the x-axis within the last 30 seconds
        if len(times_list) >= calibration_time:
            times_list.pop(0)
            medians.pop(0)


        _, bin_values, _, _ = compute_spectrogram(samples)
        _, _, median_value = calculate_signal_range(bin_values)

        medians.append(median_value)
        times_list.append(current_time)            

        plot_realtime(times_list, medians, max_value, min_value)
        print(medians)
        print(times_list)
        current_time += 1

except KeyboardInterrupt:
    print("Stopping data collection...")

sdr.pyhackrf_stop_rx()
sdr.pyhackrf_close()
