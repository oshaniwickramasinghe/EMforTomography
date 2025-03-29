import os
import numpy as np
import mmap
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import mode
from scipy.signal import spectrogram

directory_path1 = "/media/oshani/Shared/UBUNTU/EMforTomography/waru/794/diffCPU"

sampling_frequency = 20e6  
center_frequency = 794e6  
target_freq = 792e6

nfft = 2048

start_time = 5
end_time = 25

def read_iq_data(file_path):
    print(f"Reading IQ data from: {file_path}")
    with open(file_path, "rb") as f:
        print("111")
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            file_size = mm.size()
            num_samples = file_size // (2 * np.dtype(np.float32).itemsize)
            raw_data = np.frombuffer(mm, dtype=np.float32).copy()
            iq_data = raw_data[0::2] + 1j * raw_data[1::2]
            print(iq_data)
    print("Finished reading IQ data.")
    return iq_data

def extract_segment(iq_data, start_time, end_time, sampling_frequency):
    """
    Extracts a specific time segment from IQ data.
    """
    start_sample = int(start_time * sampling_frequency)
    end_sample = int(end_time * sampling_frequency)
    return iq_data[start_sample:end_sample]


def all_stat(sorted_cfile_files, directory_path):

    for cfile in sorted_cfile_files:
        file_path = os.path.join(directory_path, cfile)

        freqs, psd_accumulated = process_iq_data(file_path)

        # Adjust frequency axis to be centered at 794 MHz
        freqs = center_frequency + freqs

         # Find the index of the closest frequency bin to 792 MHz
        closest_idx = np.argmin(np.abs(freqs - target_freq))
        closest_freq = freqs[closest_idx]
        closest_psd = psd_accumulated[closest_idx]

        # plt.plot(freqs, psd_accumulated, label=f"Workload {cfile.split('_')[1].split('workload')[1]}")
        plt.plot(freqs, psd_accumulated, label=f"Workload {cfile.split('_')[1]}:{closest_psd}")
        # plt.title(f"PSD plot with and without object at worklod={cfile.split('_')[1].split('workload')[1]} %  receiver distance = 210cm, object location 30cm ")
        plt.title(f"Different CPU workloads  with object receiver distance = 210cm, object location 30cm ")

        # Plot a vertical line at 792 MHz
        # plt.axvline(x=target_freq, color='r', linestyle='--', alpha=0.5)

        # Mark the exact frequency bin
        plt.scatter(closest_freq, closest_psd, color='red', zorder=3)
        # plt.text(closest_freq, closest_psd, f"{closest_psd}", fontsize=10, color='red', verticalalignment='bottom')
        

        plt.ylim([np.min(psd_accumulated), np.min(psd_accumulated) + 15])
        print("psd_accumulated final     ", psd_accumulated)

        plt.show()


def process_iq_data(file_path, interval_duration=1):
   
    iq_data_raw = read_iq_data(file_path)
    iq_data = extract_segment(iq_data_raw, start_time, end_time, sampling_frequency)

    total_samples = len(iq_data)
    samples_per_interval = int(sampling_frequency * interval_duration)

    interval_index = 0
    psd_accumulated = None

    while interval_index * samples_per_interval < total_samples:
        start_sample = interval_index * samples_per_interval
        end_sample = start_sample + samples_per_interval

        sampling_rate = 20e6  # Example: 20 MHz


        # Extract segment and plot spectrogram
        iq_segment = iq_data[start_sample:end_sample]
        # Compute PSD using plt.psd
        fig = plt.figure()
        psd_vals, freqs = plt.psd(iq_segment, NFFT=nfft, Fs=sampling_frequency, return_line=False)

        # psd_vals = np.fft.fftshift(np.fft.fft(iq_segment))
        # n = len(iq_segment)
        # freqs = np.fft.fftshift(np.fft.fftfreq(n, d=1/sampling_rate))

        plt.close(fig)

        print("psd_accumulated   ", psd_accumulated)

        if psd_accumulated is None:
            psd_accumulated = psd_vals
        else:
            psd_accumulated += psd_vals

        interval_index += 1

    # Average the PSD across all segments
    # psd_avg = psd_accumulated / interval_index
    psd_avg_db = 10 * np.log10(psd_accumulated)

    return freqs, psd_avg_db


cfile_files = [f for f in os.listdir(directory_path1) if f.endswith('.cfile')]
sorted_cfile_files1 = sorted(cfile_files, key=lambda x: float(x.split('_')[1].split('workload')[1]))
print(sorted_cfile_files1)



plt.figure(figsize=(12, 6))


all_stat(sorted_cfile_files1, directory_path1)

plt.xlabel("Frequency (Hz)")
plt.ylabel("Power Spectral Density (dB/Hz)")
# plt.title("Averaged Power Spectral Density using plt.psd")
plt.legend()
plt.grid(True)
plt.show()