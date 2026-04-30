
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


def ode_system(t, y, b, k_in, k_f, k_1, k_2, k_3, n, c, p_1, p_2):

    (Y, X, Z_1, Z_2,
     V_Y,  V_YX,  V_YZ1, V_YZ2,
     V_X,  V_XZ1, V_XZ2,
     V_Z1, V_Z1Z2, V_Z2) = y


    dY   = c * p_1 - p_2 * Y - k_f * Y * X


    dX   = b + k_in * Y * X - k_1 * X * Z_1


    dZ_1 = k_2 * X - n * Z_1 * Z_2


    dZ_2 = k_3 - n * Z_1 * Z_2


    dV_Y = (
        p_1 * c**2
        + p_2 * Y
        - 2 * V_Y * (p_2 + k_f * X)
        + k_f * Y * X
        - 2 * k_f * Y * V_YX
    )


    dV_YX = (
        V_YX * (k_in * Y - k_1 * Z_1)
        - V_YX * (p_2 + k_f * X)
        - k_f * Y * V_X
        - k_1 * X * V_YZ1
        + k_in * X * V_Y
    )


    dV_YZ1 = (
        k_2 * V_YX
        - V_YZ1 * (p_2 + k_f * X)
        - k_f * Y * V_XZ1
        - n * Z_1 * V_YZ2
        - n * Z_2 * V_YZ1
    )


    dV_YZ2 = (
        - V_YZ2 * (p_2 + k_f * X)
        - k_f * Y * V_XZ2
        - n * Z_1 * V_YZ2
        - n * Z_2 * V_YZ1
    )


    dV_X = (
        b
        + 2 * V_X * (k_in * Y - k_1 * Z_1)
        + k_in * Y * X
        + k_1 * X * Z_1
        - 2 * k_1 * X * V_XZ1
        + 2 * k_in * X * V_YX
    )


    dV_XZ1 = (
        k_2 * V_X
        + V_XZ1 * (k_in * Y - k_1 * Z_1)
        - k_1 * X * V_Z1
        + k_in * X * V_YZ1
        - n * Z_1 * V_XZ2
        - n * Z_2 * V_XZ1
    )


    dV_XZ2 = (
        V_XZ2 * (k_in * Y - k_1 * Z_1)
        - k_1 * X * V_Z1Z2
        + k_in * X * V_YZ2
        - n * Z_1 * V_XZ2
        - n * Z_2 * V_XZ1
    )


    dV_Z1 = (
        k_2 * X
        + 2 * k_2 * V_XZ1
        + n * Z_1 * Z_2
        - 2 * n * Z_2 * V_Z1
        - 2 * n * Z_1 * V_Z1Z2
    )


    dV_Z1Z2 = (
        k_2 * V_XZ2
        + n * Z_1 * Z_2
        - n * Z_1 * V_Z2
        - n * Z_2 * V_Z1
        - n * Z_1 * V_Z1Z2
        - n * Z_2 * V_Z1Z2
    )


    dV_Z2 = (
        k_3
        + n * Z_1 * Z_2
        - 2 * n * Z_1 * V_Z2
        - 2 * n * Z_2 * V_Z1Z2
    )



    return [
        dY, dX, dZ_1, dZ_2,
        dV_Y, dV_YX, dV_YZ1, dV_YZ2,
        dV_X, dV_XZ1, dV_XZ2,
        dV_Z1, dV_Z1Z2, dV_Z2
    ]

def solve_and_save(y0, params, t_final, csv_filename):
    t_span = (0, t_final)
    t_eval = np.arange(0, t_final + 0.1, 0.1)
    sol = solve_ivp(
        fun=ode_system,
        t_span=t_span,
        y0=y0,

        args=(params['b'], params['k_in'], params['k_f'], params['k_1'], params['k_2'], params['k_3'], params['n'],
              params['c'],
              params['p_1'], params['p_2']),
        t_eval=t_eval,
        method = 'LSODA'
    )


    df = pd.DataFrame({
        't': sol.t,
        'Y': sol.y[0],
        'X': sol.y[1],
        'Z_1': sol.y[2],
        'Z_2': sol.y[3],
        'V_Y': sol.y[4],
        'V_YX': sol.y[5],
        'V_YZ1': sol.y[6],
        'V_YZ2': sol.y[7],
        'V_X': sol.y[8],
        'V_XZ1': sol.y[9],
        'V_XZ2': sol.y[10],
        'V_Z1': sol.y[11],
        'V_Z1Z2': sol.y[12],
        'V_Z2': sol.y[13]

    })

    df.to_csv(csv_filename, index=False)
    print(f"Saved solution to '{csv_filename}'")
    return df


y0 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


params = {
    'p_1': 1,
    'p_2': 1,


    'b': 25,
    'k_2': 10,
    'k_1': 1,
    'k_3': 20,
    'n': 1000,

    'c': 1,

    'k_in': 200,
    'k_f': 0.5
}


t_final = 60

t_final_ = 0

data_lna = solve_and_save(y0, params, t_final, csv_filename='lna_solution.csv')


data_lna["y_fano_lna"] = data_lna["V_Y"]/ data_lna["Y"]


v_compartment = float(get_compartments()['initial_size'].iloc[0])
avogrado_c = 6.02214076e23


data_lna["Y_part"] = [mean_i*v_compartment*avogrado_c for mean_i in data_lna["Y"]]

data_lna["Y_var_part"] = [var_i*v_compartment*avogrado_c for var_i in data_lna["V_Y"]]


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


species_names = ["y"]
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

ax1 = plt.gca()


t_plot  = np.array(t_index_controller)
y_plot  = np.array(mean_controller)
mask = t_plot >= t_final_
t_mean_controller = t_plot[mask]-t_final_
mean_controller_ = y_plot[mask]

line1, = ax1.plot(t_mean_controller, mean_controller_, linestyle="solid", color='green', label = "$Y_t\mathrm{, \, SSA}$", alpha = 0.7)

line2, = ax1.plot(data_lna["t"], data_lna["Y_part"], linestyle="dashed", color='black', label = "$Y_t\mathrm{, \, LNA}$", alpha = 0.6)

ax1.set_ylabel('Number of molecules')

plt.xlabel('Time')

plt.grid(True)
ax1.legend(handles=[line1, line2], loc = 'best')

plt.tight_layout()

plt.savefig(folder_path_exp + "/" + "y_mean_part_paper.svg", format='svg', dpi=1000)
plt.savefig(folder_path_exp + "/" + "y_mean_part_paper.pdf", format='pdf', dpi=1000)
plt.close()


plt.figure(figsize=(cm2inch(12), cm2inch(8)))


t_plot  = np.array(t_index_controller)
y_plot  = np.array(fano_controller)
mask = t_plot >= t_final_


t_index_controller_ = [float(t)-t_final_ for t, f in zip(t_plot[mask], y_plot[mask]) if f if f != "NA"]
fano_controller_ = [float(f) for f in y_plot[mask] if f if f != "NA"]


plt.plot(t_index_controller_, fano_controller_, linestyle="solid", color='green', label = "$Y_t\mathrm{, \, SSA}$", alpha = 0.7)


data_lna_ = data_lna.dropna()
plt.plot(data_lna_["t"], data_lna_["y_fano_lna"], linestyle="dashed", color='black', label = "$Y_t\mathrm{, \, LNA}$", alpha = 0.6)


b   = params["b"]
k_1 = params["k_1"]
k_2 = params["k_2"]
k_3 = params["k_3"]
k_in = params["k_in"]
k_f  = params["k_f"]

lambda_1 = (
    k_3**2 * k_f * (
        b * k_2**2 * (
            k_2 * (k_f - k_in)
            + k_3 * k_f * (k_1 + k_f - k_in)
        )
        + k_3 * k_f * (
            k_1 * k_3 * (k_2 + k_3 * k_f)
            + k_2**2 * k_in
        )
    )
)

lambda_2 = (
    (k_2 + k_3 * k_f) * (
        b * (k_2 + k_3 * k_f) * (
            k_2**2 * (b * k_2 + k_3 + k_1 * k_3**2)
            + k_2 * k_3 * (b * k_2 + 2 * k_3) * k_f
            + k_3**3 * k_f**2
        )
        + k_3**2 * k_f * (
            b * k_2**2 + k_3 * (k_2 + k_3 * k_f)
        ) * k_in
    )
)

FF_Y_t_star = 1 + (lambda_1 / lambda_2)

plt.hlines(FF_Y_t_star, min(data_lna_["t"]), max(data_lna_["t"]), color="#D55E00", alpha = 0.8, linestyle="dashdot", label = "$FF_{Y_t}^*$")


plt.ylabel('Fano factor')
plt.xlabel('Time')

plt.grid(True)
plt.legend(loc='best')
plt.ylim(None, 1)
plt.tight_layout()

plt.savefig(folder_path_exp + "/" + "y_fano_paper.svg", format='svg', dpi=1000)
plt.savefig(folder_path_exp + "/" + "y_fano_paper.pdf", format='pdf', dpi=1000)
plt.close()


import pickle
with open(folder_path_exp + f'/fano_gillespie_y_cl.pkl', 'wb') as f:
    pickle.dump({
        "t": t_index_controller_,
        "fano_y": fano_controller_,
        "t_mean": t_mean_controller,
        "mean_y": mean_controller_
    }, f)


with open(folder_path_exp + f'/fano_lna_y_cl.pkl', 'wb') as f:
    pickle.dump({
        "t": list(data_lna_["t"]),
        "fano_y": list(data_lna_["y_fano_lna"]),
        "t_mean": list(data_lna["t"]),
        "mean_y": list(data_lna["Y_part"])
    }, f)


import random
n_plots = 10
n_trajectories = 3
length = len(dict_of_states_controller["y"][0])

for plot_i in range(n_plots):
    plt.figure(figsize=(cm2inch(12), cm2inch(8)))


    sampled_trajectories = random.sample(range(length), n_trajectories)

    max_y_temp = -float('inf')
    min_y_temp = float('inf')

    colors = ["green", "black", "#D55E00"]
    for k, trajectory_i in enumerate(sampled_trajectories):

        ind_trajectory = {key: values[trajectory_i] for key, values in dict_of_states_controller["y"].items()}

        ind_trajectory_ = list(ind_trajectory.values())


        t_plot = np.array(t_index_controller)
        y_plot = np.array(ind_trajectory_)
        mask = t_plot >= t_final_

        plt.plot(t_plot[mask] - t_final_, y_plot[mask], color = colors[k], linestyle="solid", alpha=0.7)



        if max(y_plot[mask])>max_y_temp:
            max_y_temp = float(max(y_plot[mask]))
        if min(y_plot[mask])<min_y_temp:
            min_y_temp = float(min(y_plot[mask]))


    plt.ylabel('Number of molecules')

    plt.xlabel('Time')

    plt.grid(True)



    plt.tight_layout()

    plt.savefig(folder_path_exp + "/" + f"x_{n_trajectories}_trajectory_paper_{plot_i}.svg", format="svg", dpi=1000)
    plt.savefig(folder_path_exp + "/" + f"x_{n_trajectories}_trajectory_paper_{plot_i}.pdf", format="pdf", dpi=1000)

    plt.close()