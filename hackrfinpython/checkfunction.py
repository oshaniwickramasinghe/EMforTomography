#this has a problem with file type
#  https://www.youtube.com/watch?v=NrwDRkDvdkw
import argparse
import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

def record():
    # Delete existing file if it exists
    if os.path.exists("myrec"):
        os.remove("myrec")
    # Run hackrf_transfer to record data
    subprocess.run(["hackrf_transfer", "-s", "20000000", "-f", "794000000", "-r", "myrec.cfile", "-b", "20000000", "-a", "1","-l", "0", "-x", "0", "-g", "0" ])

# "-g", "1",              # Enable/disable automatic gain control (AGC) (1 = enable, 0 = disable)
# "-x", "40",             # Set RF gain (0-47 in 1 dB steps)
# "-i", "20",             # Set IF gain (0-62 in 2 dB steps)
# "-a", "1"               # Set baseband (BB) gain (0-31 in 1 dB steps)
def play():
    # Run hackrf_transfer to play recorded data
    subprocess.run(["hackrf_transfer", "-s", "20000000", "-f", "794000000", "-t", "myrec.cfile", "-a", "1", "-x", "47", "-b", "20000000"])

def plot_spectrogram():
    
    filename="myrec.cfile"
    sample_rate=8000000

    # Read the IQ data from the file
    with open(filename, "rb") as f:
        iq_data = np.fromfile(f, dtype=np.complex64)

    # Compute the spectrogram
    f, t, Sxx = spectrogram(np.real(iq_data), fs=sample_rate, nperseg=1024, noverlap=512, nfft=1024)

    # Plot the spectrogram
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='auto')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [s]')
    plt.title('Spectrogram of Recorded Data')
    plt.colorbar(label='Power [dB]')
    plt.show()

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("mode", choices=["r", "record", "p", "play", "s", "plot_spectrogram"], help="Mode of operation: 'r' or 'record' to record data, 'p' or 'play' to play recorded data")
args = parser.parse_args()

# Dispatch based on mode
if args.mode in ["r", "record"]:
    record()
elif args.mode in ["p", "play"]:
    play()
elif args.mode in ["s", "plot_spectrogram"]:
    plot_spectrogram()
else:
    print("Invalid mode. Please use 'r' or 'record' to record data, or 'p' or 'play' to play recorded data.")
