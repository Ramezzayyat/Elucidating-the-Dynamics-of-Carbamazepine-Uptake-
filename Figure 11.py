import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


data = {
    'Time (min)': [1, 30, 60, 120, 200, 300, 340, 360, 380, 400, 420, 440, 460, 480, 500, 520, 540, 560, 580, 600, 620, 640, 660, 680, 700, 720, 740, 760, 780, 800],
    'C (mg/L) 4 mL/min': [0.1, 0.1, 0.1, 0.1, 0.12, 0.18, 0.19, 0.22, 0.35, 0.38, 0.39, 0.4, 0.634, 0.752, 0.8, 1.32, 2.7, 3.22, 3.42, 3.88, 3.99, 4.1, 4.23, 4.5, 4.72, 4.72, 4.81, 4.82, 4.88, 4.88],
    'Ct/C0 4 mL/min': [0.02, 0.02, 0.02, 0.02, 0.024, 0.036, 0.038, 0.044, 0.07, 0.076, 0.078, 0.08, 0.1268, 0.1504, 0.16, 0.264, 0.54, 0.644, 0.684, 0.776, 0.798, 0.82, 0.846, 0.9, 0.944, 0.944, 0.962, 0.964, 0.976, 0.976],
    'C (mg/L) 10 mL/min': [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.25, 0.3, 0.45, 0.55, 0.6, 0.68, 0.7, 0.882, 1.224, 1.82, 3.44, 4.21, 4.5, 4.82, 4.86, 4.86, 4.87, 4.89, 4.88, 4.9, 4.9, 4.9, 4.9, 4.9],
    'Ct/C0 10 mL/min': [0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.05, 0.06, 0.09, 0.11, 0.12, 0.136, 0.14, 0.1764, 0.2448, 0.364, 0.688, 0.842, 0.9, 0.964, 0.972, 0.972, 0.974, 0.978, 0.976, 0.98, 0.98, 0.98, 0.98, 0.98],
    'C (mg/L) 15 mL/min': [0.2, 0.2, 0.2, 0.2, 0.4, 0.46, 0.5, 0.56, 0.6, 0.7, 0.89, 0.92, 1.2, 1.4, 1.52, 1.82, 3.44, 4, 4.2, 4.621, 4.7, 4.823, 4.899, 4.89, 4.89, 4.9, 4.95, 4.95, 4.95, 4.95],
    'Ct/C0 15 mL/min': [0.04, 0.04, 0.04, 0.04, 0.08, 0.092, 0.1, 0.112, 0.12, 0.14, 0.178, 0.184, 0.24, 0.28, 0.304, 0.364, 0.688, 0.8, 0.84, 0.9242, 0.94, 0.9646, 0.9798, 0.978, 0.978, 0.98, 0.99, 0.99, 0.99, 0.99]
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
plt.plot(df['Time (min)'], df['Ct/C0 4 mL/min'], 'ro-', label='Flow = 4 mL/min')
plt.plot(df['Time (min)'], df['Ct/C0 10 mL/min'], 'bs-', label='Flow = 10 mL/min')
plt.plot(df['Time (min)'], df['Ct/C0 15 mL/min'], 'g^-', label='Flow = 15 mL/min')

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