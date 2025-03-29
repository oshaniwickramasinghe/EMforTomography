#works, properly check and decide accuracy

import os
from python_hackrf import pyhackrf  
import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.signal import spectrogram
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import time

recording_time = 10  # seconds
ignore_time = 10  # seconds
center_freq = 794e6  # Hz
sample_rate = 5e6
baseband_filter = 20e6
lna_gain = 30  # 0 to 40 dB in 8 dB steps
vga_gain = 50  # 0 to 62 dB in 2 dB steps
target_freq = 792e6

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
    return times, 10 * np.log10(Sxx[target_index, :])

def plot_real_time(distances_list, medians, calibration_distances, calibration_medians, distance_range, predicted_signal):
    plt.figure(2)
    plt.clf()
    plt.plot(calibration_distances, calibration_medians, 'ro-', label='Calibration Data')
    plt.plot(distances_list, medians, 'bo-', label='Recent Data')
    plt.plot(distance_range, predicted_signal, 'b-', label='ML Regression Fit')

    plt.xlabel("Distance [m]")
    plt.ylabel("Signal Strength [dB]")
    plt.title("Real-Time Signal Strength vs Distance")
    plt.legend()
    plt.grid(True)
    plt.pause(0.01)

def train_model(distances, signal_strengths, degree=2):
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(distances, signal_strengths)
    return model

def estimate_distance_from_signal(model, signal_strength):
    """ Solve polynomial equation for distance given signal strength """
    poly_features = model.named_steps['polynomialfeatures']
    lin_reg = model.named_steps['linearregression']
    coeffs = lin_reg.coef_
    intercept = lin_reg.intercept_

    # Construct polynomial coefficients array (highest power first)
    poly_coeffs = np.zeros(len(coeffs))
    poly_coeffs[:-1] = coeffs[1:]  # Skip the first coefficient as it's for the bias
    poly_coeffs[-1] = coeffs[0] + intercept - signal_strength  # Adjust constant term

    # Solve for roots
    roots = np.roots(poly_coeffs)

    # Extract polynomial degree
    degree = len(coeffs) - 1

    # Construct equation as a string
    equation_terms = []
    for i in range(degree, 0, -1):
        equation_terms.append(f"{coeffs[i]:.4f} * d^{i}")

    # Add the constant term
    constant_term = coeffs[0] + intercept - signal_strength
    equation_terms.append(f"{constant_term:.4f}")

    # Join terms with plus signs
    equation_str = " + ".join(equation_terms)
    print(f"Polynomial equation for distance (d): {equation_str} = 0")

    file.write(f"Polynomial equation for distance (d): {equation_str} = 0")
    file.flush()

    for root in roots:
            file.write(f"\nnonfiltered = {root:.2f} m    median signal strength = {signal_strength:.2f} dB  \n")
            file.flush()
    
    # Filter real and positive roots
    valid_distances = roots[np.isreal(roots)].real
    valid_distances = valid_distances[valid_distances >= 0]

    return valid_distances


sdr.set_rx_callback(rx_callback)
sdr.pyhackrf_start_rx()

calibration_distances = np.array([0.3,0.9,1.5]).reshape(-1, 1)

calibration_medians = []
output_file = "localisation.txt"

with open(output_file, 'a') as file:
    try:
        print("Starting calibration phase...")
        for d in calibration_distances:
            print(f"Collecting calibration data at {d}m...")
            samples[:] = 0
            last_idx = 0
            time.sleep(recording_time)

            _, bin_values = compute_spectrogram(samples)
            median_value = np.median(bin_values)

            calibration_medians.append(median_value)
            print(f"Collected median signal strength: {median_value:.2f} dB")
            file.write(f"Distance = {d}    median signal strength = {median_value}\n")
            file.flush()

            print("Ignoring next 10 seconds...")
            time.sleep(ignore_time)
        
        # Train model using calibration data
        calibration_medians = np.array(calibration_medians)
        model = train_model(calibration_distances, calibration_medians)

        # Predict values for the equation obtained to plot the equation
        distance_range = np.linspace(min(calibration_distances), max(calibration_distances), 100).reshape(-1, 1)
        predicted_signal = model.predict(distance_range)

        distances_list = []
        medians = []

        while True:
            print("Collecting real-time data for 10 seconds...")
            start_time = time.time()  # Start timing

            samples[:] = 0
            last_idx = 0
            time.sleep(recording_time)

            _, bin_values = compute_spectrogram(samples)
            median_value = np.median(bin_values)

            estimated_distance = estimate_distance_from_signal(model, median_value)

            for dist in estimated_distance:
                file.write(f"estimated_distance = {dist:.2f} m    median signal strength = {median_value:.2f} dB   actual distance = \n")
                file.flush()

            # Print all valid distances
            for dist in estimated_distance:
                print(f"Estimated Distance for Signal {median_value:.2f} dB: {dist:.2f} m")
       

            if estimated_distance is not None:
                distances_list.extend(estimated_distance)
                medians.extend([median_value] * len(estimated_distance))
                plot_start_time = time.time()  # Time before plotting
                plot_real_time(distances_list, medians, calibration_distances, calibration_medians, distance_range, predicted_signal)
                plot_end_time = time.time()  # Time after plotting

                total_time = plot_end_time - start_time
                plot_time = plot_end_time - plot_start_time

                file.write(f"Total time from data collection to plot: {total_time:.3f} seconds\n")
                file.write(f"Time taken just for plotting: {plot_time:.3f} seconds\n")
                file.flush()

            else:
                print(f"Could not estimate distance for Signal {median_value:.2f} dB.")

            # if len(distances_list) >= 1:  # Only update the most recent data point
            #     distances_list.pop(0)
            #     medians.pop(0)

    except KeyboardInterrupt:
        print("Stopping data collection...")
        
sdr.pyhackrf_stop_rx()
sdr.pyhackrf_close()
