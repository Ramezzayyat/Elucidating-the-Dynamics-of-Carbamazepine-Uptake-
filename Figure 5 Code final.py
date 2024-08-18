import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

# Low concentration data (µg/L)
time_low = np.array([0, 5, 15, 30, 60, 240, 480, 1442])
qt_50 = np.array([0, 0.166, 0.183, 0.277, 0.319, 0.4, 0.41, 0.417])
qt_100 = np.array([0, 0.125, 0.204, 0.311, 0.469, 0.624, 0.67, 0.686])
qt_200 = np.array([0, 0.314, 0.461, 0.531, 0.746, 1.144, 1.264, 1.325])

# High concentration data (mg/L)
data_high = {
    'Time (min)': [0, 5, 10, 15, 20, 30, 40, 60, 90, 120, 150, 200, 400, 500, 600, 800, 1080, 1200, 1440],
    'qt (mg/g) 2 mg/L': [0, 2.8, 4.8, 6, 6.6, 6.8, 7, 7.2, 7.28, 7.4, 7.48, 7.56, 7.6, 7.652, 7.66, 7.68, 7.68, 7.68, 7.68],
    'qt (mg/g) 5 mg/L': [0, 4, 8, 10, 11.2, 11.6, 12, 12.4, 12.52, 12.72, 12.8, 13.12, 13.2, 13.36, 13.56, 13.92, 14, 14, 14],
    'qt (mg/g) 8 mg/L': [0, 2, 4, 6, 8, 8.4, 9.32, 10.52, 10.72, 11.12, 11.28, 11.6, 12, 12.32, 12.4, 12.8, 13.2, 13.2, 13.2],
    'qt (mg/g) 10 mg/L': [0, 2.4, 4, 6, 7.6, 8.8, 9.2, 9.6, 9.8, 10.68, 10.88, 11, 11.12, 11.2, 11.6, 12, 12.4, 12.4, 12.4]
}
df_high = pd.DataFrame(data_high)

# Define the pseudo-first order kinetic model
def pseudo_first_order(t, k1, qe):
    return qe * (1 - np.exp(-k1 * t))

# Define the pseudo-second order kinetic model
def pseudo_second_order(t, k2, qe):
    return (qe**2 * k2 * t) / (1 + qe * k2 * t)

# Fit and plot data
plt.figure(figsize=(15, 10))

# Pseudo-first order model for low concentrations subplot
plt.subplot(2, 2, 1)
colors = ['red', 'blue', 'green']
markers = ['o', '^', 's']
concentrations_low = ['50 µg/L', '100 µg/L', '200 µg/L']
qt_low = [qt_50, qt_100, qt_200]
k1_low = [0.03455, 0.02001, 0.01461]
qe_low_1st = [0.4071, 0.6674, 1.279]

for i, conc in enumerate(concentrations_low):
    plt.plot(time_low, qt_low[i], markers[i], label=f'{conc}', color=colors[i])
    qt_fit = pseudo_first_order(time_low, k1_low[i], qe_low_1st[i])
    plt.plot(time_low, qt_fit, linestyle='--', color=colors[i], label=f'{conc} Fit (k1={k1_low[i]}, qe={qe_low_1st[i]})')

plt.xlabel('Time (min)', fontsize=12)
plt.ylabel('qt (mg/g)', fontsize=12)
plt.legend(fontsize=10)
plt.title('a. Pseudo First Order Model (Low Concentrations)', fontsize=14)
plt.grid(True)
plt.xlim(left=0)
plt.ylim(bottom=0)

# Pseudo-second order model for low concentrations subplot
plt.subplot(2, 2, 2)
k2_low = [0.1625, 0.04398, 0.01881]
qe_low_2nd = [0.4018, 0.7051, 1.352]

for i, conc in enumerate(concentrations_low):
    plt.plot(time_low, qt_low[i], markers[i], label=f'{conc}', color=colors[i])
    qt_fit = pseudo_second_order(time_low, k2_low[i], qe_low_2nd[i])
    plt.plot(time_low, qt_fit, linestyle='-', color=colors[i], label=f'{conc} Fit (k2={k2_low[i]}, qe={qe_low_2nd[i]})')

plt.xlabel('Time (min)', fontsize=12)
plt.ylabel('qt (mg/g)', fontsize=12)
plt.legend(fontsize=10)
plt.title('c. Pseudo Second Order Model (Low Concentrations)', fontsize=14)
plt.grid(True)
plt.xlim(left=0)
plt.ylim(bottom=0)

# Pseudo-first order model for high concentrations subplot
plt.subplot(2, 2, 3)
colors_high = {'2 mg/L': 'red', '5 mg/L': 'blue', '8 mg/L': 'green', '10 mg/L': 'purple'}
markers_high = {'2 mg/L': 'o', '5 mg/L': 's', '8 mg/L': '^', '10 mg/L': 'D'}
for concentration in ['2 mg/L', '5 mg/L', '8 mg/L', '10 mg/L']:
    # Extract data
    t_high = df_high['Time (min)']
    qt_high = df_high[f'qt (mg/g) {concentration}']
    
    # Fit pseudo-first order model
    popt1, _ = curve_fit(pseudo_first_order, t_high, qt_high, bounds=(0, [1, max(qt_high)]))
    qt_fit1 = pseudo_first_order(t_high, *popt1)
    
    # Plot experimental data
    plt.plot(t_high, qt_high, markers_high[concentration], label=f'{concentration}', color=colors_high[concentration])
    
    # Plot fitted model
    plt.plot(t_high, qt_fit1, linestyle='--', color=colors_high[concentration], label=f'{concentration} PFO fit (k1={popt1[0]:.4f}, qe={popt1[1]:.4f})')

plt.xlabel('Time (min)', fontsize=12)
plt.ylabel('qt (mg/g)', fontsize=12)
plt.legend(fontsize=10)
plt.title('b. Pseudo First Order Model (High Concentrations)', fontsize=14)
plt.grid(True)
plt.xlim(left=0)
plt.ylim(bottom=0)

# Pseudo-second order model for high concentrations subplot
plt.subplot(2, 2, 4)
for concentration in ['2 mg/L', '5 mg/L', '8 mg/L', '10 mg/L']:
    # Extract data
    t_high = df_high['Time (min)']
    qt_high = df_high[f'qt (mg/g) {concentration}']
    
    # Fit pseudo-second order model
    popt2, _ = curve_fit(pseudo_second_order, t_high, qt_high, bounds=(0, [1, max(qt_high)]))
    qt_fit2 = pseudo_second_order(t_high, *popt2)
    
    # Plot experimental data
    plt.plot(t_high, qt_high, markers_high[concentration], label=f'{concentration}', color=colors_high[concentration])
    
    # Plot fitted model
    plt.plot(t_high, qt_fit2, linestyle='-', color=colors_high[concentration], label=f'{concentration} PSO fit (k2={popt2[0]:.4f}, qe={popt2[1]:.4f})')

plt.xlabel('Time (min)', fontsize=12)
plt.ylabel('qt (mg/g)', fontsize=12)
plt.legend(fontsize=10)
plt.title('d. Pseudo Second Order Model (High Concentrations)', fontsize=14)
plt.grid(True)
plt.xlim(left=0)
plt.ylim(bottom=0)

plt.tight_layout()
plt.savefig('pseudo_kinetic_models_combined.png', format='png', dpi=300)
plt.show()

# Initialize the results table
results_table = pd.DataFrame(columns=['C0 (mg/L)', 'qe exp (mg/g)', 'R2 (First Order)', 'K1 (min-1)', 'qe cal (First Order)', 'R2 (Second Order)', 'K2 (g.mg-1.min-1)', 'qe cal (Second Order)'])

# Function to calculate R2
def calculate_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

# Fit low concentration data and add to the results table
concentrations_low = [0.05, 0.1, 0.2]
qt_low = [qt_50, qt_100, qt_200]
for i, C0 in enumerate(concentrations_low):
    t = time_low
    qt = qt_low[i]
    
    # Pseudo first order model fit
    popt1, _ = curve_fit(pseudo_first_order, t, qt, bounds=(0, [1, max(qt)]))
    qt_fit1 = pseudo_first_order(t, *popt1)
    R2_1st = calculate_r2(qt, qt_fit1)
    
    # Pseudo second order model fit
    popt2, _ = curve_fit(pseudo_second_order, t, qt, bounds=(0, [1, max(qt)]))
    qt_fit2 = pseudo_second_order(t, *popt2)
    R2_2nd = calculate_r2(qt, qt_fit2)
    
    # Add to the results table
    new_row = pd.DataFrame([{
        'C0 (mg/L)': C0,
        'qe exp (mg/g)': qt[-1],
        'R2 (First Order)': R2_1st,
        'K1 (min-1)': popt1[0],
        'qe cal (First Order)': popt1[1],
        'R2 (Second Order)': R2_2nd,
        'K2 (g.mg-1.min-1)': popt2[0],
        'qe cal (Second Order)': popt2[1]
    }])
    results_table = pd.concat([results_table, new_row], ignore_index=True)

# Fit high concentration data and add to the results table
for concentration in ['2 mg/L', '5 mg/L', '8 mg/L', '10 mg/L']:
    t_high = df_high['Time (min)']
    qt_high = df_high[f'qt (mg/g) {concentration}']
    C0 = float(concentration.split(' ')[0])
    
    # Pseudo first order model fit
    popt1, _ = curve_fit(pseudo_first_order, t_high, qt_high, bounds=(0, [1, max(qt_high)]))
    qt_fit1 = pseudo_first_order(t_high, *popt1)
    R2_1st = calculate_r2(qt_high, qt_fit1)
    
    # Pseudo second order model fit
    popt2, _ = curve_fit(pseudo_second_order, t_high, qt_high, bounds=(0, [1, max(qt_high)]))
    qt_fit2 = pseudo_second_order(t_high, *popt2)
    R2_2nd = calculate_r2(qt_high, qt_fit2)
    
    # Add to the results table
    new_row = pd.DataFrame([{
        'C0 (mg/L)': C0,
        'qe exp (mg/g)': qt_high.iloc[-1],
        'R2 (First Order)': R2_1st,
        'K1 (min-1)': popt1[0],
        'qe cal (First Order)': popt1[1],
        'R2 (Second Order)': R2_2nd,
        'K2 (g.mg-1.min-1)': popt2[0],
        'qe cal (Second Order)': popt2[1]
    }])
    results_table = pd.concat([results_table, new_row], ignore_index=True)

# Print the results table
print(results_table.to_string(index=False))

# Save the table as LaTeX
latex_table = results_table.to_latex(index=False)
print(latex_table)

