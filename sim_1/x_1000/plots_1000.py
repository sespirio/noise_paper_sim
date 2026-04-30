
import sys
if '../..' not in sys.path:
    sys.path.append('../..')
from basico import *
from statistics import mean, stdev, variance
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from joblib import load
import pandas as pd
from scipy.integrate import solve_ivp
from model_controller_1000 import controller_model

matplotlib.use('TkAgg')
eps = np.finfo(np.float32).eps.item()


np.random.seed(0)



plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['DejaVu Serif']
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.useafm'] = False
plt.rcParams['pdf.use14corefonts'] = False


current_rc_params = matplotlib.rcParams.copy()


bmh_style = matplotlib.style.library['bmh']
bmh_fonts = {k: v for k, v in bmh_style.items() if 'font' in k}


matplotlib.rcParams.update(bmh_fonts)


rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']

plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 9

def cm2inch(value):
    return value/2.54

matplotlib.use('TkAgg')


current_dir = os.getcwd()
folder_name = "stochastic_simulation_paper"
folder_path_exp = os.path.join(current_dir, folder_name)
if not os.path.exists(folder_path_exp):
    os.makedirs(folder_path_exp)


model_name,species_names,species_initial_concentrations,parameters_names,parameters_initial_values,\
           reactions_names,reactions_schemes,functions_names,functions_expressions,functions_types,functions_mapping, \
           reactions_mapping,runs_stochastic,t_f,method_sim, ss_value, compartment_volume_variable \
            = controller_model()


current_dir = os.getcwd()
folder_name = "stochastic_simulation_" + model_name
folder_path = os.path.join(current_dir, folder_name)
if not os.path.exists(folder_path):
    os.makedirs(folder_path)


t_index_controller = load(folder_path + "/t_index.joblib")


model_copasi = load_model(folder_path + "/" + model_name + '.cps')


def ode_system(t, y, b, k_1, k_2, k_3, n):

    X, Z_1, Z_2, V_Z2, V_XZ2, V_Z1Z2, V_X, V_XZ1, V_Z1 = y

    dX   = b - k_1*X*Z_1

    dZ_1 = k_2*X - n*Z_1*Z_2

    dZ_2 = k_3 - n*Z_1*Z_2

    dV_Z2 = k_3 + n*Z_1*Z_2 - 2*n*Z_1*V_Z2 - 2*n*Z_2*V_Z1Z2

    dV_XZ2 = - k_1*X*V_Z1Z2 - k_1*Z_1*V_XZ2 - n*Z_1*V_XZ2 - n*Z_2*V_XZ1

    dV_Z1Z2 = k_2*V_XZ2+ n*Z_1*Z_2 - n*Z_1*V_Z2 - n*Z_2*V_Z1 - n*Z_1*V_Z1Z2 - n*Z_2*V_Z1Z2

    dV_X = b + k_1*X*Z_1 - 2*k_1*X*V_XZ1 - 2*k_1*Z_1*V_X

    dV_XZ1 = k_2*V_X - k_1*X*V_Z1 - k_1*Z_1*V_XZ1 - n*Z_1*V_XZ2 - n*Z_2*V_XZ1

    dV_Z1 = k_2*X + 2*k_2*V_XZ1 + n*Z_1*Z_2 - 2*n*Z_2*V_Z1 - 2*n*Z_1*V_Z1Z2

    return [dX, dZ_1, dZ_2, dV_Z2, dV_XZ2, dV_Z1Z2, dV_X, dV_XZ1, dV_Z1]


def solve_and_save(y0, params, t_final, n_points, csv_filename):
    t_span = (0, t_final)
    t_eval = np.arange(0, t_final + 0.1, 0.1)
    sol = solve_ivp(
        fun=ode_system,
        t_span=t_span,
        y0=y0,
        args=(params['b'], params['k_1'], params['k_2'], params['k_3'], params['n']),
        t_eval=t_eval,
        method = 'LSODA'
    )


    df = pd.DataFrame({
        't': sol.t,
        'X': sol.y[0],
        'Z_1': sol.y[1],
        'Z_2': sol.y[2],
        'V_Z2': sol.y[3],
        'V_XZ2': sol.y[4],
        'V_Z1Z2': sol.y[5],
        'V_X': sol.y[6],
        'V_XZ1': sol.y[7],
        'V_Z1': sol.y[8],

    })

    df.to_csv(csv_filename, index=False)
    print(f"Saved solution to '{csv_filename}'")
    return df


y0 = [0, 0, 0, 0, 0, 0, 0, 0, 0]


params = {
    'b': 250,
    'k_1': 1.6,
    'k_2': 1,
    'k_3': 20,
    'n': 1000
}


t_final = 15


data_lna = solve_and_save(y0, params, t_final, n_points=1000, csv_filename='lna_solution.csv')


data_lna["x_fano_lna"] = data_lna["V_X"]/ data_lna["X"]


v_compartment = float(get_compartments()['initial_size'].iloc[0])
avogrado_c = 6.02214076e23


data_lna["X_part"] = [mean_i*v_compartment*avogrado_c for mean_i in data_lna["X"]]

data_lna["X_var_part"] = [var_i*v_compartment*avogrado_c for var_i in data_lna["V_X"]]


dict_of_states_controller = {key_a: {} for key_a in species_names}


for filename in os.listdir(folder_path):
    if filename.startswith("dict_of_states_") and filename.endswith(".joblib"):

        if filename == "dict_of_states_det_conc.joblib" or filename == "dict_of_states_det.joblib":
            continue


        loaded_dict = load(os.path.join(folder_path, filename))


        for key_a in loaded_dict:
            for key_b in loaded_dict[key_a]:
                if key_b not in dict_of_states_controller[key_a]:
                    dict_of_states_controller[key_a][key_b] = []
                dict_of_states_controller[key_a][key_b].extend(loaded_dict[key_a][key_b])


species_names = ["x"]
for i, key_a in enumerate(species_names):
    mean_controller = []
    std_controller = []
    fano_controller = []
    var_controller = []

    for key_b in t_index_controller:
        list_state_at_tindex = dict_of_states_controller[key_a][key_b]
        mean_controller.append(mean(list_state_at_tindex))
        std_controller.append(stdev(list_state_at_tindex))

        mean_value = mean(list_state_at_tindex)
        variance_value = variance(list_state_at_tindex)
        var_controller.append(variance_value)

        if mean_value > 0:
            fano_factor = variance_value / mean_value
        else:
            fano_factor = "NA"

        fano_controller.append(fano_factor)


plt.figure(figsize=(cm2inch(12), cm2inch(8)))


plt.plot(t_index_controller, mean_controller, linestyle="solid", color="red", label = "Gill. [part.]", alpha = 0.6)

plt.plot(data_lna["t"], data_lna["X_part"], linestyle="solid", color="black", label = "LNA [part.]", alpha = 0.6)

plt.ylabel('Mean')
plt.xlabel('Time')

plt.grid(True)
plt.legend(loc='lower right')

plt.tight_layout()

plt.savefig(folder_path_exp + "/" + "x_mean_part_paper.svg", format='svg', dpi=1000)
plt.savefig(folder_path_exp + "/" + "x_mean_part_paper.pdf", format='pdf', dpi=1000)
plt.close()


plt.figure(figsize=(cm2inch(12), cm2inch(8)))


t_index_controller_ = [t for t, f in zip(t_index_controller, fano_controller) if f if f != "NA"]
fano_controller_ = [f for f in fano_controller if f if f != "NA"]


plt.plot(t_index_controller_, fano_controller_, linestyle="solid", color="red", label = "Gill. [part.]", alpha = 0.6)


data_lna_ = data_lna.dropna()
plt.plot(data_lna_["t"], data_lna_["x_fano_lna"], linestyle="solid", color="black", label = "LNA [conc.]", alpha = 0.6)

plt.ylabel('Fano factor')
plt.xlabel('Time')

plt.grid(True)
plt.legend(loc='lower right')

plt.tight_layout()

plt.savefig(folder_path_exp + "/" + "x_fano_paper.svg", format='svg', dpi=1000)
plt.savefig(folder_path_exp + "/" + "x_fano_paper.pdf", format='pdf', dpi=1000)
plt.close()


import random
n = 10
length = len(dict_of_states_controller["x"][0])

sampled_trajectories = random.sample(range(length), n)

for k, trajectory_i in enumerate(sampled_trajectories):
    plt.figure(figsize=(cm2inch(12), cm2inch(8)))


    ind_trajectory = {key: values[trajectory_i] for key, values in dict_of_states_controller["x"].items()}

    ind_trajectory_ = list(ind_trajectory.values())

    plt.plot(t_index_controller, ind_trajectory_, linestyle="solid", color ="red", alpha = 0.6)


    plt.ylabel('Number of molecules')

    plt.xlabel('Time')

    plt.grid(True)


    plt.tight_layout()

    plt.savefig(folder_path_exp + "/" + "x_trajectory_paper_" + str(k) + ".svg", format='svg', dpi=1000)
    plt.savefig(folder_path_exp + "/" + "x_trajectory_paper_" + str(k) +".pdf", format='pdf', dpi=1000)

    plt.close()


plt.figure(figsize=(cm2inch(12), cm2inch(8)))


plt.plot(t_index_controller, var_controller, linestyle="solid", color="red", label = "Gill. [part.]", alpha = 0.6)


plt.plot(data_lna["t"], data_lna["X_var_part"], linestyle="solid", color="black", label = "LNA [part.]", alpha = 0.6)

plt.ylabel('Variance')
plt.xlabel('Time')

plt.grid(True)
plt.legend(loc='lower right')

plt.tight_layout()

plt.savefig(folder_path_exp + "/" + "x_var_part_paper.svg", format='svg', dpi=1000)
plt.savefig(folder_path_exp + "/" + "x_var_part_paper.pdf", format='pdf', dpi=1000)
plt.close()

