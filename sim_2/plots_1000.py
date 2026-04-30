
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


def ode_system(t, y, p_1, p_2, c):

    U, V_U = y


    dU = -p_2 * U + c * p_1


    dV_U = p_1 * (c ** 2) + p_2 * U - 2 * p_2 * V_U

    return [dU, dV_U]

def solve_and_save(y0, params, t_final, csv_filename):
    t_span = (0, t_final)
    t_eval = np.arange(0, t_final + 0.1, 0.1)
    sol = solve_ivp(
        fun=ode_system,
        t_span=t_span,
        y0=y0,

        args=(params['p_1'], params['p_2'], params['c']),
        t_eval=t_eval,
        method = 'LSODA'
    )


    df = pd.DataFrame({
        't': sol.t,
        'U': sol.y[0],
        'V_U': sol.y[1]

    })

    df.to_csv(csv_filename, index=False)
    print(f"Saved solution to '{csv_filename}'")
    return df


y0 = [0, 0]


params = {
    'p_1': 0.01,
    'p_2': 0.1,
    'c': 10
}


t_final = 100
data_lna = solve_and_save(y0, params, t_final, csv_filename='lna_solution.csv')


data_lna["u_fano_lna"] = data_lna["V_U"]/ data_lna["U"]


v_compartment = float(get_compartments()['initial_size'].iloc[0])
avogrado_c = 6.02214076e23


data_lna["U_part"] = [mean_i*v_compartment*avogrado_c for mean_i in data_lna["U"]]

data_lna["U_var_part"] = [var_i*v_compartment*avogrado_c for var_i in data_lna["V_U"]]


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


species_names = ["u"]
for i, key_a in enumerate(species_names):
    mean_input = []
    std_input = []
    fano_input = []
    var_input = []

    for key_b in t_index_controller:
        list_state_at_tindex = dict_of_states_controller[key_a][key_b]
        mean_input.append(mean(list_state_at_tindex))
        std_input.append(stdev(list_state_at_tindex))

        mean_value = mean(list_state_at_tindex)
        variance_value = variance(list_state_at_tindex)
        var_input.append(variance_value)

        if mean_value > 0:
            fano_factor = variance_value / mean_value
        else:
            fano_factor = "NA"

        fano_input.append(fano_factor)

plt.figure(figsize=(cm2inch(12), cm2inch(8)))

ax1 = plt.gca()



t_plot  = np.array(t_index_controller)
y_plot  = np.array(mean_input)

line1, = ax1.plot(t_plot, y_plot, linestyle="solid", color="red", label = "Gill. [part.], u", alpha = 0.6)

line2, = ax1.plot(data_lna["t"], data_lna["U_part"], linestyle="solid", color="black", label = "LNA [part.], u", alpha = 0.6)


ax1.set_ylabel('Mean')
plt.xlabel('Time')

plt.grid(True)
ax1.legend(handles=[line1, line2])

plt.tight_layout()

plt.savefig(folder_path_exp + "/" + "u_mean_part_paper.svg", format='svg', dpi=1000)
plt.savefig(folder_path_exp + "/" + "u_mean_part_paper.pdf", format='pdf', dpi=1000)
plt.close()


plt.figure(figsize=(cm2inch(12), cm2inch(8)))

t_plot  = np.array(t_index_controller)
y_plot  = np.array(fano_input)


t_index_controller_ = [float(t) for t, f in zip(t_plot, y_plot) if f if f != "NA"]
fano_controller_ = [float(f) for f in y_plot if f if f != "NA"]


plt.plot(t_index_controller_, fano_controller_, linestyle="solid", color="red", label = "Gill. [part.]", alpha = 0.6)


data_lna_ = data_lna.dropna()
plt.plot(data_lna_["t"], data_lna_["u_fano_lna"], linestyle="solid", color="black", label = "LNA [conc.]", alpha = 0.6)

plt.ylabel('Fano factor')
plt.xlabel('Time')

plt.grid(True)
plt.legend(loc='lower right')

plt.tight_layout()

plt.savefig(folder_path_exp + "/" + "u_fano_paper.svg", format='svg', dpi=1000)
plt.savefig(folder_path_exp + "/" + "u_fano_paper.pdf", format='pdf', dpi=1000)
plt.close()


import random
n = 10
length = len(dict_of_states_controller["u"][0])

sampled_trajectories = random.sample(range(length), n)

for k, trajectory_i in enumerate(sampled_trajectories):
    plt.figure(figsize=(cm2inch(12), cm2inch(8)))


    ind_trajectory = {key: values[trajectory_i] for key, values in dict_of_states_controller["u"].items()}

    ind_trajectory_ = list(ind_trajectory.values())

    t_plot = np.array(t_index_controller)
    y_plot = np.array(ind_trajectory_)

    plt.plot(t_plot, y_plot, linestyle="solid", color ="red", alpha = 0.6)


    plt.ylabel('Number of molecules')

    plt.xlabel('Time')

    plt.grid(True)


    plt.tight_layout()

    plt.savefig(folder_path_exp + "/" + "u_trajectory_paper_" + str(k) + ".svg", format='svg', dpi=1000)
    plt.savefig(folder_path_exp + "/" + "u_trajectory_paper_" + str(k) +".pdf", format='pdf', dpi=1000)

    plt.close()

