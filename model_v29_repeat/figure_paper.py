

import sys
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
import pickle
import matplotlib.cm as cm

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



    plt.ylabel('Fano factor')
    plt.xlabel('Time')

    max_y_value = max(max(fano_controller_), max(fano_lna))
    min_y_value = min(min(fano_controller_), min(fano_lna))
    y_lower = min_y_value - abs(min_y_value) * 0.16
    y_upper = max_y_value + abs(max_y_value) * 0.16
    plt.ylim(y_lower, y_upper)

    plt.grid(True)
    plt.legend(loc='best')

    plt.tight_layout()

    plt.savefig(folder_path_exp + "/" + f"x_fano_d_{d_i}.svg", format='svg', dpi=1000)
    plt.savefig(folder_path_exp + "/" + f"x_fano_d_{d_i}.pdf", format='pdf', dpi=1000)

    plt.close()

