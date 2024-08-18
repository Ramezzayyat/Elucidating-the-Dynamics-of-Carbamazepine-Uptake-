import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Data
data = {
    'Time (min)': [0, 5, 15, 30, 60, 240, 480, 1442],
    'qt (mg/g) 50 ug/L': [0, 0.166, 0.183, 0.277, 0.319, 0.4, 0.41, 0.417],
    'qt (mg/g) 100 ug/L': [0, 0.125, 0.204, 0.311, 0.469, 0.624, 0.67, 0.686],
    'qt (mg/g) 200 ug/L': [0, 0.314, 0.461, 0.531, 0.746, 1.144, 1.264, 1.325],
    'Time (min)^0.5': [0, 2.236, 3.873, 5.477, 7.746, 15.492, 21.909, 37.974]
}

df = pd.DataFrame(data)

# Define the function to plot linear regression and extend lines
def plot_extended_linear_regression(x, y, color, extend_left=None, extend_right=None):
    # Reshape x for sklearn
    x_reshaped = x.values.reshape(-1, 1)
    
    # Create and fit the model
    model = LinearRegression()
    model.fit(x_reshaped, y)
    
    # Predict y values
    x_extended = np.linspace(extend_left if extend_left else x.min(), extend_right if extend_right else x.max(), 100).reshape(-1, 1)
    y_pred = model.predict(x_extended)
    
    # Plot the regression line
    plt.plot(x_extended, y_pred, color=color, linestyle='--')

# Segment the data into two regions
threshold = 8 

# Plot data and linear regression for each segment
plt.figure(figsize=(10, 6))

colors = {'50 ug/L': 'red', '100 ug/L': 'blue', '200 ug/L': 'green'}
markers = {'50 ug/L': 'o', '100 ug/L': 's', '200 ug/L': '^'}

for concentration in ['50 ug/L', '100 ug/L', '200 ug/L']:
    segment_1 = df[(df['Time (min)^0.5'] <= threshold) & (df[f'qt (mg/g) {concentration}'] > 0)]
    segment_2 = df[(df['Time (min)^0.5'] > threshold) & (df[f'qt (mg/g) {concentration}'] > 0)]
    
    # Plot original data
    plt.plot(df['Time (min)^0.5'][df[f'qt (mg/g) {concentration}'] > 0], 
             df[f'qt (mg/g) {concentration}'][df[f'qt (mg/g) {concentration}'] > 0], 
             markers[concentration], label=f'{concentration} ', color=colors[concentration])

    # Plot extended regression lines
    plot_extended_linear_regression(segment_1['Time (min)^0.5'], segment_1[f'qt (mg/g) {concentration}'], colors[concentration], extend_right=15)
    plot_extended_linear_regression(segment_2['Time (min)^0.5'], segment_2[f'qt (mg/g) {concentration}'], colors[concentration], extend_left=8)

# Format plot
plt.xlabel('time$^{0.5}$ (min$^{0.5}$)', fontsize=12)
plt.ylabel('$q_t$ (mg/g)', fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.xlim(left=0)
plt.ylim(bottom=0)

# Save plot with regression lines with 300 dpi
plt.savefig('adsorption_kinetics_with_segmented_regression.png', dpi=300)
plt.show()
