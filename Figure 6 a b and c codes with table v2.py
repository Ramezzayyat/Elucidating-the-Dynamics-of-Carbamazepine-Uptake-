import pandas as pd
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Load the dataset
file_path = r"C:\\Users\\rz29\\OneDrive - American University of Beirut\\Documents\\Reem's paper\\Journal of Industrial and Engineering Chemistry\\Data for model 4.csv"
data = pd.read_csv(file_path)

# Constants
V = 0.1  # Volume of the solution in liters
m = 0.025  # Mass of the adsorbent in grams

# Initial CBZ concentrations for each experiment
initial_concentrations = {
    'Experiment 1': 2,
    'Experiment 2': 5,
    'Experiment 3': 8,
    'Experiment 4': 10,
    'Experiment 5': 2,
    'Experiment 6': 5,
    'Experiment 7': 8,
    'Experiment 8': 2,
    'Experiment 9': 5
}

# Calculate qt for each data point
data['qt (mg/g)'] = data.apply(lambda row: ((initial_concentrations[row['Experiment Identifier']] - row['Remaining CBZ Concentration (mg/L)']) * V) / m, axis=1)

# Estimating qe (equilibrium adsorption capacity) for each experiment
qe_estimates = data.groupby('Experiment Identifier')['qt (mg/g)'].max()

# Function to represent the LDF model (differential equation)
def ldf_model(qt, t, kL, qe):
    return kL * (qe - qt)

# Function to calculate the sum of squares of the residuals
def ldf_residuals(kL, time, qt, qe):
    qt_model = odeint(ldf_model, 0, time, args=(kL, qe)).flatten()
    return sum((qt - qt_model) ** 2)

# Optimization process for each experiment
optimized_kL_values = {}

for experiment in qe_estimates.index:
    # Select data for the experiment and its qe estimate
    exp_data = data[data['Experiment Identifier'] == experiment]
    qe_exp = qe_estimates[experiment]

    # Time and qt values for fitting
    time_exp = exp_data['Time (minutes)'].values
    qt_exp = exp_data['qt (mg/g)'].values

    # Perform optimization to find the best kL
    result_exp = minimize(ldf_residuals, 0.01, args=(time_exp, qt_exp, qe_exp), method='Nelder-Mead')
    kL_optimized_exp = result_exp.x[0]
    optimized_kL_values[experiment] = kL_optimized_exp

# Create a table to store the results
results_table = pd.DataFrame(columns=['Experiment', 'R2', 'RSS', 'kL_optimized'])

# Calculate R2, RSS, and kL_optimized for each experiment
residuals_data = []

for experiment in qe_estimates.index:
    # Select data for the experiment and its optimized kL and qe
    exp_data = data[data['Experiment Identifier'] == experiment]
    qe_exp = qe_estimates[experiment]
    kL_optimized_exp = optimized_kL_values[experiment]

    # Time and qt values for the experiment
    time_exp = exp_data['Time (minutes)'].values
    qt_exp = exp_data['qt (mg/g)'].values

    # Solving the LDF model with optimized kL
    qt_model = odeint(ldf_model, 0, time_exp, args=(kL_optimized_exp, qe_exp)).flatten()

    # Calculate residuals
    residuals = qt_exp - qt_model
    residuals_data.append(pd.DataFrame({'Experiment': experiment, 'Time (minutes)': time_exp, 'Residuals': residuals}))

    # Calculate R2
    ss_total = sum((qt_exp - np.mean(qt_exp)) ** 2)
    ss_residual = sum((qt_exp - qt_model) ** 2)
    r2 = 1 - (ss_residual / ss_total)

    # Calculate RSS
    rss = sum((qt_exp - qt_model) ** 2)

    # Add the results to the table
    new_row = pd.DataFrame([{'Experiment': experiment, 'R2': r2, 'RSS': rss, 'kL_optimized': kL_optimized_exp}])
    results_table = pd.concat([results_table, new_row], ignore_index=True)

# Print the results table
print(results_table)

# Plot residuals for each experiment
plt.figure(figsize=(15, 10))

for i, experiment in enumerate(qe_estimates.index, 1):
    residuals_exp = pd.concat(residuals_data).loc[pd.concat(residuals_data)['Experiment'] == experiment]
    plt.subplot(3, 3, i)
    plt.plot(residuals_exp['Time (minutes)'], residuals_exp['Residuals'], 'o-')
    plt.title(f'Residuals for {experiment}', fontsize=12)
    plt.xlabel('Time (minutes)', fontsize=10)
    plt.ylabel('Residuals (mg/g)', fontsize=10)
    plt.grid(True)

plt.tight_layout()
plt.show()

# Combined plots with similar pH values in one figure
unique_ph_values = data['pH'].unique()
fig, axs = plt.subplots(1, len(unique_ph_values), figsize=(20, 6))

for idx, ph in enumerate(unique_ph_values):
    ax = axs[idx]
    ph_data = data[data['pH'] == ph]
    for experiment in ph_data['Experiment Identifier'].unique():
        exp_data = ph_data[ph_data['Experiment Identifier'] == experiment]
        qe_exp = qe_estimates[experiment]
        kL_optimized_exp = optimized_kL_values[experiment]
        time_exp = exp_data['Time (minutes)'].values
        qt_exp = exp_data['qt (mg/g)'].values
        qt_model = odeint(ldf_model, 0, time_exp, args=(kL_optimized_exp, qe_exp)).flatten()
        initial_concentration_exp = initial_concentrations[experiment]
        ax.plot(time_exp, qt_exp, 'o', label=f'Initial Concentration {initial_concentration_exp} mg/L', markersize=5)
        ax.plot(time_exp, qt_model, '--', label=f'LDF Model (Initial Concentration {initial_concentration_exp} mg/L, kL={kL_optimized_exp:.4f})')
    ax.set_title(f'pH {ph}', fontsize=14)
    ax.set_xlabel('Time (min)', fontsize=12)
    ax.set_ylabel('qt (mg/g)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

# Save the combined figure with 300 dpi
plt.tight_layout()
plt.savefig('combined_figure.png', dpi=300)
plt.show()

# Convert the results_table DataFrame to a LaTeX table
latex_table = results_table.to_latex(index=False)

# Print the LaTeX table
print(latex_table)
