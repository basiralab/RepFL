import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import random

import sys
import os

from constants import *
from utils.utils import *
from plots import *

# Add the parent directory to the sys.path
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)


if torch.cuda.is_available():
    device = torch.device("cuda")
    print("running on GPU")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("running on GPU MPS")
else:
    device = torch.device("cpu")
    print("running on CPU")

manual_seed = 1773
random.seed(manual_seed)
np.random.seed(manual_seed)
torch.manual_seed(manual_seed)

create_directories_for_saving()

# Add names of the runs here (the directory where models + results are saved)
# Observation: the script reads the mae test values from files which are created when running test/test_one_method_slim.py to plot the results
run_names = []

methods = ['Standalone', 'FedAvg', 'FedDyn', 'FedDC', 'RepFL']
index_to_client = {'0': 'H1', '1': 'H2', '2': 'H3', '3': 'H4'}


# ************   Resolution 160   ************

mae_all = {}

for client in range(2):
    for method in methods:
        mae_all[str(client) + method] = []

for method, run_name in zip(methods, run_names):
    for fold in range(NUMBER_FOLDS):
        mae_current_fold = read_mae_from_file(fold, run_name)
        for client in range(2):
            mae_all[str(client) + method].append(mae_current_fold[client])

data = []
for client in range(2):
    for method in methods:
        values = mae_all[str(client) + method]
        for val in values:
            val = val.tolist()
            print(val)
            data.append([index_to_client[str(client)], method, val])

# Create a DataFrame
columns = ['Client', 'Method', 'Run']
df = pd.DataFrame(data, columns=columns)

# Create the bar plot using Seaborn
plot_colours = ['turquoise', 'purple', 'palevioletred', 'coral', 'deepskyblue']
sns.set_style("whitegrid")

plt.figure(figsize=(10, 6))
ax = sns.barplot(data=df, x='Client', y='Run', hue='Method', ci='sd', palette=plot_colours, alpha=0.8)
# ax.legend_.remove()  # Remove the legend
plt.ylim([0.163, 0.185])

plt.xlabel('Hospital')
plt.ylabel('MAE')

plt.savefig(f'{SAVING_FOLDER_PATH}/{GLOBAL_RUN_NAME}/average_runs_{160}_mae.png')

plt.close()




# ************   Resolution 268   ************

mae_all = {}

for client in range(2, NUMBER_CLIENTS):
    for method in methods:
        mae_all[str(client) + method] = []

for method, run_name in zip(methods, run_names):
    for fold in range(NUMBER_FOLDS):
        mae_current_fold = read_mae_from_file(fold, run_name)
        for client in range(2, NUMBER_CLIENTS):
            mae_all[str(client) + method].append(mae_current_fold[client])

data = []
for client in range(2, NUMBER_CLIENTS):
    for method in methods:
        values = mae_all[str(client) + method]
        for val in values:
            val = val.tolist()
            print(val)
            data.append([index_to_client[str(client)], method, val])

# Create a DataFrame
columns = ['Client', 'Method', 'Run']
df = pd.DataFrame(data, columns=columns)

sns.set_style("whitegrid")

plt.figure(figsize=(10, 6))
ax = sns.barplot(data=df, x='Client', y='Run', hue='Method', ci='sd', palette=plot_colours, alpha=0.8)
# ax.legend_.remove()  # Remove the legend
plt.ylim([0.183, 0.23])

plt.xlabel('Hospital')
plt.ylabel('MAE')

plt.savefig(f'{SAVING_FOLDER_PATH}/{GLOBAL_RUN_NAME}/average_runs_{268}_mae.png')

plt.close()