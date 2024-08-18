import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Data for dV/dW Pore Volume vs. Pore Size
pore_size = [
    1.48329, 1.59052, 1.71561, 1.85858, 2.00155, 2.16239, 2.3411, 2.5198, 2.73426, 2.94871,
    3.18103, 3.43122, 3.69929, 4.00309, 4.32477, 4.66432, 5.03961, 5.43277, 5.87954, 6.34419,
    6.84458, 7.39858, 7.98832, 8.63167, 9.31077, 10.0613, 10.8655
]
dv_dw_pore_volume = [
    0, 0.0844557, 0.085082, 0.112748, 0.0913675, 0.0813248, 0.0647887, 0.0661928, 0.0685315, 0.050865,
    0.0390987, 0.0494564, 0.0431712, 0.0232111, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
]

# Create a figure with a 1x2 grid layout for the two plots side by side
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Plot dV/dW Pore Volume vs. Pore Size on the left
axs[0].plot(pore_size, dv_dw_pore_volume, marker='s', color='green', linestyle='-', linewidth=2, markersize=6)
axs[0].set_title('a. Pore Volume vs. Pore Size')
axs[0].set_xlabel('Pore Size (nm)')
axs[0].set_ylabel('dV/dW Pore Volume (cm³/g·nm)')
axs[0].grid(True)

# Annotate the peaks with red "x" marks and show only the x-value
peaks = [1.85, 2.72, 3.45]
for peak in peaks:
    idx = (np.abs(np.array(pore_size) - peak)).argmin()
    axs[0].plot(pore_size[idx], dv_dw_pore_volume[idx], "x", color='red')
    axs[0].annotate(f'{pore_size[idx]:.2f}', 
                    xy=(pore_size[idx], dv_dw_pore_volume[idx]),
                    textcoords="offset points", xytext=(0,10), ha='center',
                    fontsize=10)

# Load the XRD data
file_path = r"C:\Users\rz29\OneDrive - American University of Beirut\Documents\Reem's paper\CBZ BET reports\XRD on DPAC\pit date activated carbon_exported.xy"
xrd_data = pd.read_csv(file_path, sep='\s+', header=None, skiprows=1)
xrd_data = xrd_data.iloc[:, [0, 1]]
xrd_data.columns = ['2Theta', 'Intensity']

# Ensure that the data is read as floats
xrd_data['2Theta'] = pd.to_numeric(xrd_data['2Theta'], errors='coerce')
xrd_data['Intensity'] = pd.to_numeric(xrd_data['Intensity'], errors='coerce')
xrd_data = xrd_data.dropna()

# Plot the XRD pattern on the right
axs[1].plot(xrd_data['2Theta'], xrd_data['Intensity'], color='blue', linewidth=1)
axs[1].set_title('b. XRD Pattern of DPAC')
axs[1].set_xlabel('2θ (Degrees)')
axs[1].set_ylabel('Intensity (a.u.)')
axs[1].grid(True)

# Annotate specific peaks for 26.7 and 43.68 with red "x" marks and show only the x-value
specific_peaks = [26.7, 43.68]
for peak in specific_peaks:
    idx = (np.abs(xrd_data['2Theta'] - peak)).argmin()
    axs[1].plot(xrd_data['2Theta'][idx], xrd_data['Intensity'][idx], "x", color='red')
    axs[1].annotate(f'{xrd_data["2Theta"][idx]:.2f}', 
                    xy=(xrd_data['2Theta'][idx], xrd_data['Intensity'][idx]), 
                    textcoords="offset points", xytext=(0,10), ha='center')

# Adjust layout to prevent overlapping and optimize spacing
plt.tight_layout()

# Save the figure with 300 dpi resolution
plt.savefig('combined_pore_xrd_analysis_side_by_side.png', dpi=600)

# Show the combined figure
plt.show()
