
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
folder_name = "tailored_figs_paper"
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


def ode_system(t, y, b, k_1, k_2, k_3, n,
               d):

    X, Z_1, Z_2, V_Z2, V_XZ2, V_Z1Z2, V_X, V_XZ1, V_Z1 = y


    dX = b - k_1 * X * Z_1 - d * X


    dZ_1 = k_2 * X - n * Z_1 * Z_2


    dZ_2 = k_3 - n * Z_1 * Z_2


    dV_X =  d * X + b + k_1 * X * Z_1 - 2 * k_1 * X * V_XZ1 - 2 * (k_1 * Z_1 + d) * V_X


    dV_XZ1 = k_2 * V_X - k_1 * X * V_Z1 - (k_1 * Z_1 + d) * V_XZ1 - n * Z_1 * V_XZ2 - n * Z_2 * V_XZ1


    dV_XZ2 = - k_1 * X * V_Z1Z2 - (k_1 * Z_1 + d) * V_XZ2 - n * Z_1 * V_XZ2 - n * Z_2 * V_XZ1


    dV_Z1 = k_2 * X + 2 * k_2 * V_XZ1 + n * Z_1 * Z_2 - 2 * n * Z_2 * V_Z1 - 2 * n * Z_1 * V_Z1Z2


    dV_Z1Z2 = k_2 * V_XZ2 + n * Z_1 * Z_2 - n * Z_1 * V_Z2 - n * Z_2 * V_Z1 - n * Z_1 * V_Z1Z2 - n * Z_2 * V_Z1Z2


    dV_Z2 = k_3 + n * Z_1 * Z_2 - 2 * n * Z_1 * V_Z2 - 2 * n * Z_2 * V_Z1Z2

    return [dX, dZ_1, dZ_2, dV_Z2, dV_XZ2, dV_Z1Z2, dV_X, dV_XZ1, dV_Z1]


def solve_and_save(y0, params, t_final, csv_filename):
    t_span = (0, t_final)
    t_eval = np.arange(0, t_final + 0.1, 0.1)
    sol = solve_ivp(
        fun=ode_system,
        t_span=t_span,
        y0=y0,
        args=(params['b'], params['k_1'], params['k_2'], params['k_3'], params['n'],
              params['d']),
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


def ode_system_new(t, y, b, k_in, k_1, k_2, k_3, n,
                   d,
                   c, p_1, p_2):

    U, X, Z_1, Z_2, V_U, V_UX, V_UZ1, V_UZ2, V_X, V_XZ1, V_XZ2, V_Z1, V_Z1Z2, V_Z2 = y


    dU   = c * p_1 - p_2 * U


    dX   = b + k_in * U * X - k_1 * X * Z_1 - d * X


    dZ_1 = k_2 * X - n * Z_1 * Z_2


    dZ_2 = k_3     - n * Z_1 * Z_2


    dV_U  = p_1 * c**2 + p_2 * U - 2 * p_2 * V_U


    dV_UX =  - (d - k_in * U + k_1 * Z_1) * V_UX - p_2 * V_UX - k_1 * X * V_UZ1 + k_in * X * V_U


    dV_UZ1 = k_2 * V_UX - p_2 * V_UZ1 - n * Z_1 * V_UZ2 - n * Z_2 * V_UZ1


    dV_UZ2 =  - p_2 * V_UZ2 - n * Z_1 * V_UZ2 - n * Z_2 * V_UZ1


    dV_X =  d * X + b - 2 * V_X * (d - k_in * U + k_1 * Z_1) + k_in * U * X + k_1 * X * Z_1 - 2 * k_1 * X * V_XZ1 + 2 * k_in * X * V_UX


    dV_XZ1 =  k_2 * V_X - (d - k_in * U + k_1 * Z_1) * V_XZ1 - k_1 * X * V_Z1 + k_in * X * V_UZ1 - n * Z_1 * V_XZ2 - n * Z_2 * V_XZ1


    dV_XZ2 = - (d - k_in * U + k_1 * Z_1) * V_XZ2 - k_1 * X * V_Z1Z2 + k_in * X * V_UZ2 - n * Z_1 * V_XZ2 - n * Z_2 * V_XZ1


    dV_Z1 =  k_2 * X + 2 * k_2 * V_XZ1 + n * Z_1 * Z_2 - 2 * n * Z_2 * V_Z1 - 2 * n * Z_1 * V_Z1Z2


    dV_Z1Z2 =  k_2 * V_XZ2 + n * Z_1 * Z_2 - n * Z_1 * V_Z2 - n * Z_2 * V_Z1 - n * Z_1 * V_Z1Z2 - n * Z_2 * V_Z1Z2


    dV_Z2 =  k_3 + n * Z_1 * Z_2 - 2 * n * Z_1 * V_Z2 - 2 * n * Z_2 * V_Z1Z2


    return [dU, dX, dZ_1, dZ_2, dV_U, dV_UX, dV_UZ1, dV_UZ2, dV_X, dV_XZ1, dV_XZ2, dV_Z1, dV_Z1Z2, dV_Z2]

def solve_and_save_new(y0_new, params_new, t_final_new, csv_filename):
    t_span = (0, t_final_new)
    t_eval = np.arange(0, t_final_new + 0.1, 0.1)
    sol = solve_ivp(
        fun=ode_system_new,
        t_span=t_span,
        y0=y0_new,

        args=(params_new['b'], params_new['k_in'], params_new['k_1'], params_new['k_2'], params_new['k_3'], params_new['n'],
              params_new['d'],
              params_new['c'], params_new['p_1'], params_new['p_2']),
        t_eval=t_eval,
        method = 'LSODA'
    )


    df = pd.DataFrame({
        't': sol.t,
        'U': sol.y[0],
        'X': sol.y[1],
        'Z_1': sol.y[2],
        'Z_2': sol.y[3],
        'V_U': sol.y[4],
        'V_UX': sol.y[5],
        'V_UZ1': sol.y[6],
        'V_UZ2': sol.y[7],
        'V_X': sol.y[8],
        'V_XZ1': sol.y[9],
        'V_XZ2': sol.y[10],
        'V_Z1': sol.y[11],
        'V_Z1Z2': sol.y[12],
        'V_Z2': sol.y[13],

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
    'n': 1000,

    'd': 0.1
}


t_final = 15

data_lna = solve_and_save(y0, params, t_final, csv_filename='lna_solution.csv')


X_f = float(data_lna[data_lna.columns[1]].iloc[-1])
Z_1_f = float(data_lna[data_lna.columns[2]].iloc[-1])
Z_2_f = float(data_lna[data_lna.columns[3]].iloc[-1])
V_Z2_f = float(data_lna[data_lna.columns[4]].iloc[-1])
V_XZ2_f = float(data_lna[data_lna.columns[5]].iloc[-1])
V_Z1Z2_f = float(data_lna[data_lna.columns[6]].iloc[-1])
V_X_f = float(data_lna[data_lna.columns[7]].iloc[-1])
V_XZ1_f = float(data_lna[data_lna.columns[8]].iloc[-1])
V_Z1_f = float(data_lna[data_lna.columns[9]].iloc[-1])


y0_new = [0, X_f, Z_1_f, Z_2_f, 0, 0, 0, 0, V_X_f, V_XZ1_f, V_XZ2_f, V_Z1_f, V_Z1Z2_f, V_Z2_f]


params_new = {
    'b': 250,
    'k_1': 1.6,
    'k_2': 1,
    'k_3': 20,
    'n': 1000,

    'd': 0.1,

    'k_in': 1.6,
    'p_1': 0.01,
    'p_2': 0.1,
    'c': 10
}


t_final_new = 100-t_final

data_lna_new = solve_and_save_new(y0_new, params_new, t_final_new, csv_filename='lna_solution_new.csv')


data_lna_new["x_fano_lna"] = data_lna_new["V_X"]/ data_lna_new["X"]
data_lna_new["u_fano_lna"] = data_lna_new["V_U"]/ data_lna_new["U"]


v_compartment = float(get_compartments()['initial_size'].iloc[0])
avogrado_c = 6.02214076e23


data_lna_new["X_part"] = [mean_i*v_compartment*avogrado_c for mean_i in data_lna_new["X"]]
data_lna_new["U_part"] = [mean_i*v_compartment*avogrado_c for mean_i in data_lna_new["U"]]

data_lna_new["X_var_part"] = [var_i*v_compartment*avogrado_c for var_i in data_lna_new["V_X"]]
data_lna_new["U_var_part"] = [var_i*v_compartment*avogrado_c for var_i in data_lna_new["V_U"]]


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

plt.figure(figsize=(cm2inch(12), cm2inch(6)))

ax1 = plt.gca()




t_plot  = np.array(t_index_controller)
y_plot  = np.array(mean_controller)
mask = t_plot >= t_final
line1, = ax1.plot(t_plot[mask]-t_final, y_plot[mask], linestyle="solid", color="green", label = "$X\mathrm{, \, SSA}$", alpha = 0.7)

line2, = ax1.plot(data_lna_new["t"], data_lna_new["X_part"], linestyle="dashed", color="black", label = "$X\mathrm{, \, LNA}$", alpha = 0.6)


output_ss = float(list(ss_value.values())[0])
du_dt_gill = np.gradient(np.asarray(mean_input), t_plot) + output_ss



mask = t_plot >= t_final
line3, = ax1.plot(t_plot[mask] - t_final, du_dt_gill[mask], linestyle="dashdot", color="#D55E00", label = "$\mathrm{d}U / \mathrm{d}t \mathrm{, \, SSA}$", alpha = 0.6)



ax1.set_ylabel('Number of molecules')

plt.xlabel('Time')

plt.grid(True)
ax1.legend(handles=[line1, line2, line3])


max_y_value = max(max(y_plot[mask]), max(du_dt_gill[mask]))
min_y_value = min(min(y_plot[mask]), min(du_dt_gill[mask]))
y_lower = min_y_value - abs(min_y_value) * 0.007
y_upper = max_y_value + abs(max_y_value) * 0.009
plt.ylim(y_lower, y_upper)

plt.tight_layout()

plt.savefig(folder_path_exp + "/" + "x_mean_part_paper.svg", format='svg', dpi=1000)
plt.savefig(folder_path_exp + "/" + "x_mean_part_paper.pdf", format='pdf', dpi=1000)
plt.close()



import random
n_plots = 10
n_trajectories = 3
length = len(dict_of_states_controller["x"][0])

for plot_i in range(n_plots):
    plt.figure(figsize=(cm2inch(12), cm2inch(8)))

    sampled_trajectories = random.sample(range(length), n_trajectories)

    max_y_temp = -float('inf')
    min_y_temp = float('inf')

    colors = ["green", "black", "#D55E00"]
    for k, trajectory_i in enumerate(sampled_trajectories):

        ind_trajectory = {key: values[trajectory_i] for key, values in dict_of_states_controller["x"].items()}

        ind_trajectory_ = list(ind_trajectory.values())

        t_plot = np.array(t_index_controller)
        y_plot = np.array(ind_trajectory_)
        mask = t_plot >= t_final

        plt.plot(t_plot[mask] - t_final, y_plot[mask], color = colors[k], linestyle="solid", alpha=0.7)



        if max(y_plot[mask])>max_y_temp:
            max_y_temp = float(max(y_plot[mask]))
        if min(y_plot[mask])<min_y_temp:
            min_y_temp = float(min(y_plot[mask]))


    plt.ylabel('Number of molecules')

    plt.xlabel('Time')

    plt.grid(True)



    y_lower = min_y_temp - abs(min_y_temp) * 0.1
    y_upper = max_y_temp + abs(max_y_temp) * 0.1


    plt.tight_layout()

    plt.savefig(folder_path_exp + "/" + f"x_{n_trajectories}_trajectory_paper_{plot_i}.svg", format="svg", dpi=1000)
    plt.savefig(folder_path_exp + "/" + f"x_{n_trajectories}_trajectory_paper_{plot_i}.pdf", format="pdf", dpi=1000)

    plt.close()