import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

data = {
    'Time (min)': [1, 30, 60, 120, 200, 300, 340, 360, 380, 400, 420, 440, 460, 480, 500, 520, 540, 560, 580, 600, 620, 640, 660, 680, 700, 720, 740, 760, 780, 800, 820, 840, 860, 880, 900, 920, 940, 960, 980, 1000, 1020, 1030],
    'C (mg/L) 2mg/L': [0.01, 0.015, 0.016, 0.016, 0.032, 0.033, 0.035, 0.044, 0.049, 0.05, 0.06, 0.09, 0.1, 0.115, 0.119, 0.18, 0.21, 0.34, 0.39, 0.42, 0.53, 0.709, 0.8, 0.9, 0.9556, 1.02, 1.09, 1.3, 1.4, 1.45, 1.6, 1.823, 1.854, 1.868, 1.899, 1.922, 1.942, 1.9448, 1.958, 1.967, 1.989, 1.989],
    'Ct/C0 2mg/L': [0.005, 0.0075, 0.008, 0.008, 0.016, 0.0165, 0.0175, 0.022, 0.0245, 0.025, 0.03, 0.045, 0.05, 0.0575, 0.0595, 0.09, 0.105, 0.17, 0.195, 0.21, 0.265, 0.3545, 0.4, 0.45, 0.4778, 0.51, 0.545, 0.65, 0.7, 0.725, 0.8, 0.9115, 0.927, 0.934, 0.9495, 0.961, 0.971, 0.9724, 0.979, 0.9835, 0.9945, 0.9945],
    'C (mg/L) 5mg/L': [0.1, 0.1, 0.1, 0.1, 0.12, 0.18, 0.19, 0.22, 0.35, 0.38, 0.39, 0.4, 0.634, 0.752, 0.8, 1.32, 2.7, 3.22, 3.42, 3.88, 3.99, 4.1, 4.23, 4.5, 4.72, 4.72, 4.81, 4.82, 4.88, 4.88, 4.88, 4.72, 4.81, 4.82, 4.88, 4.88, 4.88, 4.81, 4.88, 4.88, 4.88, 4.88],
    'Ct/C0 5mg/L': [0.02, 0.02, 0.02, 0.02, 0.024, 0.036, 0.038, 0.044, 0.07, 0.076, 0.078, 0.08, 0.1268, 0.1504, 0.16, 0.264, 0.54, 0.644, 0.684, 0.776, 0.798, 0.82, 0.846, 0.9, 0.944, 0.944, 0.962, 0.964, 0.976, 0.976, 0.976, 0.944, 0.962, 0.964, 0.976, 0.976, 0.976, 0.962, 0.976, 0.976, 0.976, 0.976],
    'C (mg/L) 10mg/L': [0.2, 0.2, 0.8, 1.3, 1.5, 2.8, 3, 3.4, 3.843, 4, 6.05, 9.42, 9.45, 9.49, 9.51, 9.522, 9.5633, 9.6, 9.671, 9.754, 9.88, 9.88, 9.88, 9.88, 9.88, 9.88, 9.88, 9.88, 9.88, 9.88, 9.88, 9.88, 9.88, 9.88, 9.88, 9.88, 9.88, 9.88, 9.88, 9.88, 9.88, 9.88, 9.88, 9.88, 9.88],
    'Ct/C0 10mg/L': [0.02, 0.02, 0.08, 0.13, 0.15, 0.28, 0.3, 0.34, 0.3843, 0.4, 0.605, 0.9, 0.945, 0.949, 0.949, 0.951, 0.9522, 0.95633, 0.96, 0.9671, 0.9754, 0.988, 0.988, 0.988, 0.988, 0.988, 0.988, 0.988, 0.988, 0.988, 0.988, 0.988, 0.988, 0.988, 0.988, 0.988, 0.988, 0.988, 0.988, 0.988, 0.988, 0.988, 0.988]
}

# Find the minimum length of all arrays
min_length = min(len(v) for v in data.values())

# Truncate all arrays to the minimum length
for key in data:
    data[key] = data[key][:min_length]

# Create DataFrame
df = pd.DataFrame(data)



# Plotting the breakthrough curves
plt.figure(figsize=(10, 6))
plt.plot(df['Time (min)'], df['Ct/C0 2mg/L'], 'ro-', label='C0 = 2mg/L')
plt.plot(df['Time (min)'], df['Ct/C0 5mg/L'], 'bs-', label='C0 = 5mg/L')
plt.plot(df['Time (min)'], df['Ct/C0 10mg/L'], 'g^-', label='C0 = 10mg/L')

# Add markers for Ct/C0 = 0.5 and 1.0
plt.axhline(y=0.5, color='red', linestyle='--', linewidth=0.8)
plt.axhline(y=1.0, color='red', linestyle='--', linewidth=0.8)


plt.xlabel('Time (min)', fontsize=12)
plt.ylabel('Ct/C0', fontsize= 12)
plt.legend(fontsize=10)
plt.grid(True)
plt.tight_layout()
plt.xlim(left=0)
plt.ylim(bottom=0)
plt.savefig('Breakthrouh1.png', dpi=300)
plt.show()