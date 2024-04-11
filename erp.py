import numpy as np
import matplotlib.pyplot as plt
import scipy.io

# Load data from the file S00.mat
mat_data = scipy.io.loadmat('S00.mat')

# Extract the first row of X, Y, and I
X_first_row = mat_data['X']
Y_first_row = mat_data['Y'][0]
I_first_row = mat_data['I'][0]

# Define the event codes (Y values) for ERPs
event_codes = [3, 4]

# Set values in Y outside the event codes to 0
Y_events = np.where(np.isin(Y_first_row, event_codes), Y_first_row, 0)

# Find the indices where Y equals the event codes
event_indices = np.where(Y_events > 0)[0]

# Define the time window around each event (in seconds)
time_window_before = 0.2  # 200 milliseconds before event
time_window_after = 0.8   # 800 milliseconds after event

# Extract the EEG readings (X) for each event within the time window for all channels
erps = []
for event_index in event_indices:
    # Calculate the time range around the event
    event_time = I_first_row[event_index]
    window_start = event_time - time_window_before
    window_end = event_time + time_window_after
    
    # Find the indices within the time window
    window_indices = np.where((I_first_row >= window_start) & (I_first_row <= window_end))[0]
    
    # Extract EEG readings for this event for all channels
    if X_first_row.ndim == 1:
        event_eeg = X_first_row[window_indices]
    else:
        event_eeg = X_first_row[:, window_indices]
    
    # Append the event EEG readings to the list of ERPs
    erps.append(event_eeg)

# Convert the list of ERPs to a numpy array for further analysis
erps = np.array(erps)

# Randomly select # indices for plotting
num_plots = 20
random_indices = np.random.choice(len(erps), num_plots, replace=False)

# Calculate the time points for the ERP waveform
time_points = np.linspace(-time_window_before, time_window_after, erps.shape[2])

# Plot the average ERP waveform across all channels for the selected indices
plt.figure(figsize=(12, 8))
for i, index in enumerate(random_indices):
    plt.subplot(4, 5, i+1)
    plt.plot(time_points, erps[index].mean(axis=0))
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (uV)')
    plt.title(f'ERP for Event {index}')
    plt.grid(True)

plt.tight_layout()
plt.show()
