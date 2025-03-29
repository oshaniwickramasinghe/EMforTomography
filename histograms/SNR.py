import os
import numpy as np
import mmap
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

def read_iq_data(file_path):
    with open(file_path, "rb") as f:
        print(f"Reading IQ data from: {file_path}")
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            file_size = mm.size()
            num_samples = file_size // (2 * np.dtype(np.float32).itemsize)
            raw_data = np.frombuffer(mm, dtype=np.float32).copy()
            iq_data = raw_data[0::2] + 1j * raw_data[1::2]
    print("Finished reading IQ data.")
    return iq_data

def compute_spectrogram(iq_segment, sampling_frequency, center_frequency, target_freq):
    """If you are comparing power levels at a specific frequency (like in your case), scaling='spectrum' is usually better, because it gives the actual signal power rather than power per Hz.

🔸 If you need to compare across different frequency resolutions or estimate noise power spectral density (PSD), then scaling='density' is more appropriate."""
    frequencies, times, Sxx = spectrogram(
        iq_segment,
        fs=sampling_frequency,
        window='hann',
        nperseg=1024,
        noverlap=512,
        nfft=2048,
        scaling='spectrum'
    )
    frequencies_shifted = frequencies + (center_frequency - sampling_frequency / 2)
    target_index = np.abs(frequencies_shifted - target_freq).argmin()
    return times, 10 * np.log10(Sxx[target_index, :])  # Convert power to dB

def compute_snr(signal_power, noise_power):
    return 10 * np.log10(signal_power / noise_power)

def process_iq_data(file_path, sampling_frequency, center_frequency, target_freq, interval_duration=1):
    iq_data = read_iq_data(file_path)
    total_samples = len(iq_data)
    samples_per_interval = int(sampling_frequency * interval_duration)
    temp_target_bin_segments = np.array([])
    temp_time = np.array([])
    interval_index = 0
    while interval_index * samples_per_interval < total_samples:
        start_sample = interval_index * samples_per_interval
        end_sample = start_sample + samples_per_interval
        iq_segment = iq_data[start_sample:end_sample]
        target_time, target_bin_segments = compute_spectrogram(iq_segment, sampling_frequency, center_frequency, target_freq)
        temp_target_bin_segments = np.concatenate((temp_target_bin_segments, target_bin_segments))
        temp_time = np.concatenate((temp_time, target_time))
        interval_index += 1
    return temp_time, temp_target_bin_segments


# Define paths for with-object, without-object, and noise reference
# directory_with_object = "/media/oshani/Shared/UBUNTU/EMforTomography/waru/794/waruat1feet/use"
directory_with_object = "/media/oshani/Shared/UBUNTU/EMforTomography/waru/794/movingwaruantennaat7"
directory_without_object = "/media/oshani/Shared/UBUNTU/EMforTomography/waru/794/movingwaruantennaat7/nouse"
directory_noise = "/media/oshani/Shared/UBUNTU/EMforTomography/waru/794/desktopoff/use"


# directory_with_object = "/media/oshani/Shared/UBUNTU/EMforTomography/ground/sanduni"
# directory_without_object = "/media/oshani/Shared/UBUNTU/EMforTomography/ground/none"
# directory_noise = "/media/oshani/Shared/UBUNTU/EMforTomography/ground/off"


# Define parameters
sampling_frequency = 20e6  # 20 MHz
center_frequency = 794e6   # 794 MHz
target_freq = 792e6       # 792 MHz

# Get sorted file lists
def get_sorted_cfile_list(directory):
    cfile_files = [f for f in os.listdir(directory) if f.endswith('.cfile')]
    return sorted(cfile_files, key=lambda x: float(x.split('_')[1].split('f')[0]))


sorted_cfile_files1 = get_sorted_cfile_list(directory_with_object)
sorted_cfile_files2 = get_sorted_cfile_list(directory_without_object)
sorted_cfile_files3 = get_sorted_cfile_list(directory_noise)

sorted_cfile_files2 = ['30s_7f_WOWend.cfile','30s_7f_WOWend.cfile','30s_7f_WOWend.cfile','30s_7f_WOWend.cfile','30s_7f_WOWend.cfile','30s_7f_WOWend.cfile','30s_7f_WOWend.cfile']
sorted_cfile_files3 = ['30s_7f.cfile','30s_7f.cfile','30s_7f.cfile','30s_7f.cfile','30s_7f.cfile','30s_7f.cfile','30s_7f.cfile']


print(sorted_cfile_files1)
print(sorted_cfile_files2)
print(sorted_cfile_files3)

# Process and analyze the data
def process_directory(sorted_files1, sorted_files2, sorted_files3, 
                      directory1, directory2, directory3, output_file="SNR_values_movingobject.txt"):

    snr_with_list = []
    snr_without_list = []
    
    with open(output_file, 'a') as file:  # Open in append mode
        for cfile1, cfile2, cfile3 in zip(sorted_files1, sorted_files2, sorted_files3):
            file_with_object = os.path.join(directory1, cfile1)
            file_without_object = os.path.join(directory2, cfile2)
            file_noise = os.path.join(directory3, cfile3)

            # Process data
            time_obj, power_with_object = process_iq_data(file_with_object, sampling_frequency, center_frequency, target_freq)
            time_no_obj, power_without_object = process_iq_data(file_without_object, sampling_frequency, center_frequency, target_freq)
            time_noise, power_noise = process_iq_data(file_noise, sampling_frequency, center_frequency, target_freq)

            # Convert power from dB to linear scale
            power_with_object_linear = 10 ** (power_with_object / 10)
            power_without_object_linear = 10 ** (power_without_object / 10)
            power_noise_linear = 10 ** (power_noise / 10)

            # Compute SNR
            snr_with_object = compute_snr(power_with_object_linear.mean(), power_noise_linear.mean())
            snr_without_object = compute_snr(power_without_object_linear.mean(), power_noise_linear.mean())

            # Compute SNR difference
            snr_difference = snr_without_object - snr_with_object

            # Append results for plotting
            snr_with_list.append(snr_with_object)
            snr_without_list.append(snr_without_object)

            # Write to file
            file.write(f"SNR Without Object: {file_without_object}__{snr_without_object:.2f} dB\n")
            file.write(f"SNR With Object: {file_with_object}__{snr_with_object:.2f} dB\n")
            file.write(f"ΔSNR (Impact of Object): {file_noise}__{snr_difference:.2f} dB\n")
            file.write("-" * 40 + "\n")

    # Plot results
    plt.figure(figsize=(10, 5))
    plt.bar(['Without Object', 'With Object'], [np.mean(snr_without_list), np.mean(snr_with_list)], color=['green', 'red'])
    plt.ylabel("SNR (dB)")
    plt.title("Average SNR Comparison With and Without Object")
    plt.grid(True)
    plt.show()

# Call function
process_directory(sorted_cfile_files1, sorted_cfile_files2, sorted_cfile_files3, 
                  directory_with_object, directory_without_object, directory_noise)
