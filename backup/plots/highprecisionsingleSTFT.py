#this code works with precision of values upto 50 decimals
import numpy as np
import mmap
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from decimal import Decimal, getcontext

# Set the precision for Decimal
getcontext().prec = 50  # Adjust precision as needed

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
    # Convert In and Qn to float first before passing to Decimal
    In = Decimal(float(In))
    Qn = Decimal(float(Qn))

    # Perform calculations with Decimal for higher precision
    real_part = In * Decimal(np.cos(2 * np.pi * f * n)) - Qn * Decimal(np.sin(2 * np.pi * f * n))
    # print('real', n, real_part)
    imag_part = Qn * Decimal(np.cos(2 * np.pi * f * n)) + In * Decimal(np.sin(2 * np.pi * f * n))
    # print('imag', n, imag_part)

    return real_part, imag_part

def process_chunks(chunks, f):
    g = 0
    all_results = []
    for chunk in chunks:
        print('n of chunk', g, chunk[0])
        for n in range(len(chunk)):
            In = chunk[n].real
            Qn = chunk[n].imag
            
            # Use compute_xn to maintain higher precision
            returned_IQ = compute_xn(In, Qn, f, n, 20000000)
            # print('n of chunk', n, returned_IQ)

            # Store the result (real_part, imag_part) in the list
            all_results.append(returned_IQ)
        g += 1
    return all_results

# Path to the IQ data file
file_path1 = "./chunks/01tile800MNOstrs10MB.cfile"
IQlist1 = read_iq_data(file_path1)

# Process IQ data
allchunks = read_in_chunks(IQlist1)
processed_chunks = process_chunks(allchunks, 0.1)
# print(processed_chunks[0])

# Plotly graph visualization (unchanged)
import plotly.graph_objects as go

processed_data = np.concatenate(processed_chunks)
real_processed_data = np.abs(processed_data)
# print(real_processed_data[:50])

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
    title="Processed IQ Data Over Time f=20000000",
    xaxis=dict(
        title="Sample Index",
        rangeslider=dict(visible=True),  # Enables the scrollable slider
    ),
    yaxis=dict(title="Processed Value (Power)")
)

fig.show()
