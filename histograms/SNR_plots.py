# import matplotlib.pyplot as plt
# import re

# def parse_snr_data(file_path):
#     distance = []
#     snr_without_object = []
#     snr_with_object = []
#     delta_snr = []
    
#     with open(file_path, 'r') as file:
#         for line in file:
#             match = re.search(r'30s_(\d+)f.*__([-\d\.]+) dB', line)
#             # match = re.search(r'30s_([\d\.]+)m.*__([-\d\.]+) dB', line)
#             if match:
#                 freq = float(match.group(1))
#                 snr_value = float(match.group(2))
                
#                 if 'Without Object' in line:
#                     distance.append(freq)
#                     snr_without_object.append(snr_value)
#                 elif 'With Object' in line:
#                     snr_with_object.append(snr_value)
#                 elif '\u0394SNR' in line:  # Delta SNR line
#                     delta_snr.append(snr_value)
    
#     return distance, snr_without_object, snr_with_object, delta_snr

# def plot_snr(distance, snr_without, snr_with, delta_snr):
#     plt.figure(figsize=(10, 5))
    
#     # Plot the lines
#     plt.plot(distance, snr_without, marker='o', label='SNR Without Object', linestyle='--',color='orange')
#     plt.plot(distance, snr_with, marker='o', label='SNR With Object', linestyle='--',color='dodgerblue')
#     plt.plot(distance, delta_snr, marker='o', label='ΔSNR (Impact of Object)', linestyle='--',color='gray')

#     # Add text annotations for each point
#     for i, d in enumerate(distance):
#         plt.text(d, snr_without[i], f'{snr_without[i]:.2f}', fontsize=9, ha='right', va='bottom',color='orange')
#         plt.text(d, snr_with[i], f'{snr_with[i]:.2f}', fontsize=9, ha='left', va='bottom',color='dodgerblue')
#         plt.text(d, delta_snr[i], f'{delta_snr[i]:.2f}', fontsize=9, ha='center', va='top',color='gray')

#     # Labels and grid
#     plt.xlabel('Distance (m)')
#     plt.ylabel('SNR (dB)')
#     plt.title('SNR Analysis for area of the link')
#     plt.legend()
#     plt.grid()
    
#     # Show the plot
#     plt.show()


# # File path (update with your actual file location)
# file_path = '/media/oshani/Shared/UBUNTU/EMforTomography/SNR_values_movingobject.txt'  # Replace with the actual filename

# # Parse the data
# distance, snr_without, snr_with, delta_snr = parse_snr_data(file_path)

# # Plot the data
# plot_snr(distance, snr_without, snr_with, delta_snr)



import matplotlib.pyplot as plt
import re

def parse_snr_data(file_path):
    distance = []
    snr_without_object = []
    snr_with_object = []
    delta_snr = []
    
    with open(file_path, 'r') as file:
        for line in file:
            match = re.search(r'30s_(\d+)f.*__([-\d\.]+) dB', line)
            if match:
                freq = float(match.group(1))
                snr_value = float(match.group(2))
                
                if 'Without Object' in line:
                    distance.append(freq)
                    snr_without_object.append(snr_value)
                elif 'With Object' in line:
                    snr_with_object.append(snr_value)
                elif '\u0394SNR' in line:  # Delta SNR line
                    delta_snr.append(snr_value)
    
    return distance, snr_without_object, snr_with_object, delta_snr

def plot_snr(distance, snr_without, snr_with, delta_snr, threshold_snr=None):
    plt.figure(figsize=(10, 5))

    distance = [d * 0.3 for d in distance]
    
    # Plot the lines
    plt.plot(distance, snr_without, marker='o', label='SNR Without Object', linestyle='--', color='orange')
    plt.plot(distance, snr_with, marker='o', label='SNR With Object', linestyle='--', color='dodgerblue')
    plt.plot(distance, delta_snr, marker='o', label='ΔSNR (Impact of Object)', linestyle='--', color='gray')

    # Add text annotations for each point
    for i, d in enumerate(distance):
        plt.text(d, snr_without[i], f'{snr_without[i]:.2f}', fontsize=9, ha='right', va='bottom', color='orange')
        plt.text(d, snr_with[i], f'{snr_with[i]:.2f}', fontsize=9, ha='left', va='bottom', color='dodgerblue')
        plt.text(d, delta_snr[i], f'{delta_snr[i]:.2f}', fontsize=9, ha='center', va='top', color='gray')

    plt.xticks(distance, [f'{d:.1f}' for d in distance])
    
    # If a threshold SNR is given, add a horizontal line and mark intersections
    if threshold_snr is not None:
        plt.axhline(y=threshold_snr, color='red', linestyle='-', label=f'SNR Threshold = {threshold_snr} dB')
        
        # Find the x-values where the threshold line intersects the curves
        for label, snr_data, color in [('Without Object', snr_without, 'orange'), 
                                       ('With Object', snr_with, 'dodgerblue'), 
                                       ('ΔSNR', delta_snr, 'gray')]:
            for i in range(1, len(distance)):
                # Check for intersection
                if (snr_data[i-1] < threshold_snr and snr_data[i] >= threshold_snr) or \
                   (snr_data[i-1] > threshold_snr and snr_data[i] <= threshold_snr):
                    # Estimate the x-coordinate of intersection
                    x_intersect = distance[i-1] + (distance[i] - distance[i-1]) * (threshold_snr - snr_data[i-1]) / (snr_data[i] - snr_data[i-1])
                    plt.text(x_intersect, threshold_snr, f'{x_intersect:.2f} m', fontsize=9, ha='left', va='bottom', color=color)

    # Labels and grid
    plt.xlabel('Distance (m)')
    # xticks = plt.gca().get_xticks()
    # plt.xticks(xticks, [1*float(x) for x in xticks])
    plt.ylabel('SNR (dB)')
    plt.title('SNR Analysis for Area of the Link')
    plt.legend()
    plt.grid()
    
    # Show the plot
    plt.show()

# File path (update with your actual file location)
file_path = '/media/oshani/Shared/UBUNTU/EMforTomography/SNR_values_movingobject.txt'  # Replace with the actual filename

# Parse the data
distance, snr_without, snr_with, delta_snr = parse_snr_data(file_path)

# Set the SNR threshold value you want to plot
threshold_snr = 2.145  # For example, -5 dB

# Plot the data with the threshold line
plot_snr(distance, snr_without, snr_with, delta_snr, threshold_snr)
