#this code is working with a truncation error in compute_xn when creating the comlex value 
import numpy as np
import mmap
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Function to read IQ data from file
def read_iq_data(file_path):
    with open(file_path, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            file_size = mm.size()
            num_samples = file_size // (2 * np.dtype(np.float32).itemsize)
            print(num_samples)

            raw_data = np.frombuffer(mm, dtype=np.float32).copy()

            iq_data = raw_data[0::2] + 1j * raw_data[1::2]  # Create complex IQ data

    return iq_data
    
def read_in_chunks(data, chunk_size=2000):
    chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
    print(len(chunks))
    return chunks

def compute_xn(In, Qn, f, n, N):
    real_part = In * np.cos(2 * np.pi * f * n) - Qn * np.sin(2 * np.pi * f * n)
    print('real',n,real_part)
    imag_part = Qn * np.cos(2 * np.pi * f * n) + In * np.sin(2 * np.pi * f * n)
    print('imag',n,imag_part)
    return complex(real_part, imag_part)

# def compute_xn(In, Qn, f, n, N):
#     real_part = In * np.cos((2 * np.pi * f * n) / N) - Qn * np.sin((2 * np.pi * f * n) / N)
#     print('real',n,real_part)
#     imag_part = Qn * np.cos((2 * np.pi * f * n) / N) + In * np.sin((2 * np.pi * f * n) / N)
#     print('imag',n,imag_part)
#     return complex(real_part, imag_part)


def process_chunks(chunks, f):
    g=0
    for chunk in chunks:
        # print('before',g,chunk[:2])
        for n in range(len(chunk)):
            In = chunk[n].real
            Qn = chunk[n].imag
            
            chunk[n] = compute_xn(In, Qn, f, n,20000000)
            # chunk[n] = np.square(chunk[n].real) + np.square(chunk[n].imag)
            print('n of chunk',n,chunk[n])
        # print('after',g,chunk[:2])
        g+=1
    return chunks


# Path to the IQ data file
file_path1 = "./chunks/01tile800MNOstrs10MB.cfile"
IQlist1 = read_iq_data(file_path1)

# Process IQ data
allchunks = read_in_chunks(IQlist1)
processed_chunks = process_chunks(allchunks,0.79)
print(processed_chunks[0])

# # iterative line graph

# # Set up the plot for animation
# fig, ax = plt.subplots()
# line, = ax.plot([], [], lw=2)
# ax.set_xlim(0, len(processed_chunks[0])/2)
# ax.set_ylim(0, max(map(max, processed_chunks)))

# def init():
#     line.set_data([], [])
#     return line,

# def update(frame):
#     y_data = processed_chunks[frame]
#     x_data = np.arange(len(y_data))
#     line.set_data(x_data, y_data)
#     return line,

# # Animate the plot, showing each chunk sequentially
# ani = FuncAnimation(fig, update, frames=len(processed_chunks), init_func=init, blit=True)

# plt.show()


#one line graph

# Aggregate all processed chunks into a single array
processed_data = np.concatenate(processed_chunks)

# Plot the processed data
plt.figure(figsize=(12, 6))
plt.plot(processed_data, color='blue', linewidth=0.1)
plt.title("Processed IQ Data Over Time for 2MHz")
plt.xlabel("Sample Index")
plt.ylabel("Processed Value (Power)")
plt.show()

import plotly.graph_objects as go

processed_data = np.concatenate(processed_chunks)
real_processed_data = np.abs(processed_data)
print(real_processed_data[:50])

# Create a Plotly figure
fig = go.Figure()

# Add trace for processed data
fig.add_trace(go.Scatter(
    x=np.arange(len(real_processed_data)),
    y=real_processed_data,
    mode='lines',
    line=dict(color='blue', width=0.2)
))

# Customize layout for interactive scrolling
fig.update_layout(
    title="Processed IQ Data Over Time for 2MHz using 0.1 as f",
    xaxis=dict(
        title="Sample Index",
        rangeslider=dict(visible=True),  # Enables the scrollable slider
    ),
    yaxis=dict(title="Processed Value (Power)")
)

fig.show()


