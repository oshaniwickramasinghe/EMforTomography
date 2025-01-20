import numpy as np
import mmap
import plotly.graph_objects as go
from scipy.signal import spectrogram

# Function to read IQ data
def read_iq_data(file_path):
    print("Start reading data")
    with open(file_path, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            file_size = mm.size()
            raw_data = np.frombuffer(mm, dtype=np.float32).copy()
            iq_data = raw_data[0::2] + 1j * raw_data[1::2]
    print("Finished reading data")
    return iq_data

# Function to save filtered IQ data
def save_filtered_data(iq_data, file_path):
    print("Saving filtered data")
    real_part = np.real(iq_data).astype(np.float32)
    imag_part = np.imag(iq_data).astype(np.float32)
    interleaved_data = np.empty((real_part.size + imag_part.size,), dtype=np.float32)
    interleaved_data[0::2] = real_part
    interleaved_data[1::2] = imag_part
    with open(file_path, "wb") as f:
        interleaved_data.tofile(f)
    print(f"Filtered data saved to {file_path}")

# Function to remove unwanted segments
def remove_segments(iq_data, sampling_frequency, intervals_to_exclude):
    total_samples = len(iq_data)
    keep_mask = np.ones(total_samples, dtype=bool)

    for start_time, end_time in intervals_to_exclude:
        start_sample = int(start_time * sampling_frequency)
        end_sample = int(end_time * sampling_frequency)
        keep_mask[start_sample:end_sample] = False

    filtered_data = iq_data[keep_mask]
    print("Unwanted segments removed")
    return filtered_data

# Main processing
intervals_to_exclude = [
    (4, 6)  # Exclude data from 2s to 3s
    # (5.1, 5.5)   # Exclude data from 7s to 8s
]


file_name = "794_3t_null.cfile"
iq_data_file = f"/media/oshani/Shared/UBUNTU/EMforTomography/794/no_object/{file_name}"
saved_file_name = "filtered_794_3t_null.cfile"


# # Parameters
sampling_frequency = 20e6  # 20 MHz

iq_data = read_iq_data(iq_data_file)
filtered_iq_data = remove_segments(iq_data, sampling_frequency, intervals_to_exclude)
save_filtered_data(filtered_iq_data, saved_file_name)
