
import sys
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
import pickle


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


d_values = [0.1, 0.5, 1, 2.5, 5, 7.5]


with open(f'1000_d_{0}/stochastic_simulation_paper/fano_gillespie_d_{0}.pkl', 'rb') as f:
    gill_data = pickle.load(f)
    t_index_controller_0 = gill_data["t"]
    fano_controller_0 = gill_data["fano_x"]


with open(f'1000_d_{0}/stochastic_simulation_paper/fano_lna_d_{0}.pkl', 'rb') as f:
    lna_data = pickle.load(f)
    t_lna_0 = lna_data["t"]
    fano_lna_0 = lna_data["fano_x"]


for idx, d_i in enumerate(d_values):

    plt.figure(figsize=(cm2inch(11.4), cm2inch(6)))




    with open(f'1000_d_{d_i}/stochastic_simulation_paper/fano_gillespie_d_{d_i}.pkl', 'rb') as f:
        gill_data = pickle.load(f)
        t_index_controller_ = gill_data["t"]
        fano_controller_ = gill_data["fano_x"]


    with open(f'1000_d_{d_i}/stochastic_simulation_paper/fano_lna_d_{d_i}.pkl', 'rb') as f:
        lna_data = pickle.load(f)
        t_lna = lna_data["t"]
        fano_lna = lna_data["fano_x"]


    plt.plot(t_index_controller_, fano_controller_, linestyle="solid", color='green', label = "$X\mathrm{, \, SSA}$", alpha = 0.7)


    plt.plot(t_lna, fano_lna, linestyle="dashed", color='black', label = "$X\mathrm{, \, LNA}$", alpha = 0.6)


    params_new = {
        'b': 250,
        'k_1': 1.6,
        'k_2': 1,
        'k_3': 20,
        'n': 1000,



        'k_in': 32,
        'p_1': 0.01,
        'p_2': 0.1,
        'c': 10
    }



    FF_x = FF_X_star = 1 + (
            params_new["k_3"] * params_new["p_2"] * (
            2 * (params_new["k_1"] ** 2) * (params_new["k_3"] ** 3)
            + params_new["c"] * (1 + params_new["c"]) * (params_new["k_2"] ** 2) * (params_new["k_in"] ** 2) *
            params_new["p_1"]
            + 2 * params_new["k_1"] * params_new["k_3"] * (
                    params_new["c"] * params_new["k_2"] * params_new["k_in"] * params_new["p_1"]
                    + params_new["p_2"] * (params_new["b"] * params_new["k_2"] + params_new["k_3"] * params_new["p_2"])
            )
    )
    ) / (
                               2 * (params_new["k_2"] ** 2) * (
                                   params_new["c"] * params_new["k_in"] * params_new["p_1"] + params_new["b"] *
                                   params_new["p_2"])
                               * (
                                       params_new["k_1"] * (params_new["k_3"] ** 2)
                                       + params_new["c"] * params_new["k_2"] * params_new["k_in"] * params_new["p_1"]
                                       + params_new["p_2"] * (
                                                   params_new["b"] * params_new["k_2"] + params_new["k_3"] * params_new[
                                               "p_2"])
                               )
                       )
    plt.hlines(FF_x, min(t_lna_0), max(t_lna_0), color="#D55E00", alpha=0.8, linestyle="dashdot", label="$FF_X^*$")

    plt.ylabel('Fano factor')
    plt.xlabel('Time')

    plt.ylim(2.75, 4.25)

    plt.grid(True)
    plt.legend(loc='upper right')

    plt.tight_layout()

    plt.savefig(folder_path_exp + "/" + f"x_fano_d_{d_i}.svg", format='svg', dpi=1000)
    plt.savefig(folder_path_exp + "/" + f"x_fano_d_{d_i}.pdf", format='pdf', dpi=1000)

    plt.close()
