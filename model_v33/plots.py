
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


current_dir = os.getcwd()
folder_name = "stochastic_simulation_"
folder_path = os.path.join(current_dir, folder_name)
if not os.path.exists(folder_path):
    os.makedirs(folder_path)


def ode_system(t, y, b, k_in, k_1, k_2, k_3, n, c, p_1, p_2):


    (U, X, Z_1, Z_2,
     V_U, V_UX, V_UZ1, V_UZ2,
     V_X, V_XZ1, V_XZ2,
     V_Z1, V_Z1Z2, V_Z2) = y


    dU = c * p_1 - p_2 * U


    dX = b + k_in * U - k_1 * X * Z_1


    dZ_1 = k_2 * X - n * Z_1 * Z_2


    dZ_2 = k_3 - n * Z_1 * Z_2


    dV_U = p_1 * c**2 + p_2 * U - 2 * p_2 * V_U


    dV_UX = k_in * V_U \
            - p_2 * V_UX \
            - k_1 * X * V_UZ1 \
            - k_1 * Z_1 * V_UX


    dV_UZ1 = k_2 * V_UX \
             - p_2 * V_UZ1 \
             - n * Z_1 * V_UZ2 \
             - n * Z_2 * V_UZ1


    dV_UZ2 = - p_2 * V_UZ2 \
             - n * Z_1 * V_UZ2 \
             - n * Z_2 * V_UZ1


    dV_X = b + k_in * U + 2 * k_in * V_UX \
           + k_1 * X * Z_1 \
           - 2 * k_1 * X * V_XZ1 \
           - 2 * k_1 * Z_1 * V_X


    dV_XZ1 = k_2 * V_X \
             + k_in * V_UZ1 \
             - k_1 * X * V_Z1 \
             - k_1 * Z_1 * V_XZ1 \
             - n * Z_1 * V_XZ2 \
             - n * Z_2 * V_XZ1


    dV_XZ2 = k_in * V_UZ2 \
             - k_1 * X * V_Z1Z2 \
             - k_1 * Z_1 * V_XZ2 \
             - n * Z_1 * V_XZ2 \
             - n * Z_2 * V_XZ1


    dV_Z1 = k_2 * X \
            + 2 * k_2 * V_XZ1 \
            + n * Z_1 * Z_2 \
            - 2 * n * Z_2 * V_Z1 \
            - 2 * n * Z_1 * V_Z1Z2


    dV_Z1Z2 = k_2 * V_XZ2 \
              + n * Z_1 * Z_2 \
              - n * Z_1 * V_Z2 \
              - n * Z_2 * V_Z1 \
              - n * Z_1 * V_Z1Z2 \
              - n * Z_2 * V_Z1Z2


    dV_Z2 = k_3 \
            + n * Z_1 * Z_2 \
            - 2 * n * Z_1 * V_Z2 \
            - 2 * n * Z_2 * V_Z1Z2


    return [
        dU, dX, dZ_1, dZ_2,
        dV_U, dV_UX, dV_UZ1, dV_UZ2,
        dV_X, dV_XZ1, dV_XZ2,
        dV_Z1, dV_Z1Z2, dV_Z2
    ]


def solve_and_save(y0, params, t_final=85, csv_filename=None):
    t_span = (0, t_final)
    t_eval = np.arange(0, t_final + 0.1, 0.1)
    sol = solve_ivp(
        fun=ode_system,
        t_span=t_span,
        y0=y0,
        args=(params['b'], params['k_in'], params['k_1'], params['k_2'], params['k_3'], params['n'], params['c'],
              params['p_1'], params['p_2']),
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
        'V_Z2': sol.y[13]

    })

    df.to_csv(csv_filename, index=False)
    print(f"Saved solution to '{csv_filename}'")
    return df


def ode_system_new(t, y,
                             p_2, k_1, k_2, k_in, n,
                             Z1_star, Z2_star, X_star,
                             V_UZ1_star, V_UZ2_star, V_UX_star,
                             V_XZ1_star, V_XZ2_star, V_X_star,
                             V_Z1_star, V_Z2_star, V_Z1Z2_star):


    (u, x, z_1, z_2,
     v_u, v_ux, v_uz1, v_uz2,
     v_x, v_xz1, v_xz2,
     v_z1, v_z1z2, v_z2) = y


    du = -p_2 * u


    dx = -k_1 * Z1_star * x - k_1 * X_star * z_1 + k_in * u


    dz_1 = k_2 * x - n * Z2_star * z_1 - n * Z1_star * z_2


    dz_2 = -n * Z2_star * z_1 - n * Z1_star * z_2


    dv_u = p_2 * u - 2 * p_2 * v_u


    dv_ux = (k_in * v_u
             - p_2 * v_ux
             - k_1 * X_star * v_uz1
             - k_1 * V_UZ1_star * x
             - k_1 * Z1_star * v_ux
             - k_1 * V_UX_star * z_1)


    dv_uz1 = (k_2 * v_ux
              - p_2 * v_uz1
              - n * Z1_star * v_uz2
              - n * V_UZ2_star * z_1
              - n * Z2_star * v_uz1
              - n * V_UZ1_star * z_2)


    dv_uz2 = (-p_2 * v_uz2
              - n * Z1_star * v_uz2
              - n * V_UZ2_star * z_1
              - n * Z2_star * v_uz1
              - n * V_UZ1_star * z_2)


    dv_x = (k_in * u
            + 2 * k_in * v_ux
            + k_1 * Z1_star * x
            + k_1 * X_star * z_1
            - 2 * k_1 * X_star * v_xz1
            - 2 * k_1 * V_XZ1_star * x
            - 2 * k_1 * Z1_star * v_x
            - 2 * k_1 * V_X_star * z_1)


    dv_xz1 = (k_2 * v_x
              + k_in * v_uz1
              - k_1 * Z1_star * v_xz1
              - k_1 * V_XZ1_star * z_1
              - k_1 * X_star * v_z1
              - k_1 * V_Z1_star * x
              - n * Z1_star * v_xz2
              - n * V_XZ2_star * z_1
              - n * Z2_star * v_xz1
              - n * V_XZ1_star * z_2)


    dv_xz2 = (k_in * v_uz2
              - k_1 * X_star * v_z1z2
              - k_1 * V_Z1Z2_star * x
              - (k_1 + n) * Z1_star * v_xz2
              - (k_1 + n) * V_XZ2_star * z_1
              - n * Z2_star * v_xz1
              - n * V_XZ1_star * z_2)


    dv_z1 = (k_2 * x
             + 2 * k_2 * v_xz1
             + n * Z2_star * z_1
             + n * Z1_star * z_2
             - 2 * n * Z2_star * v_z1
             - 2 * n * V_Z1_star * z_2
             - 2 * n * Z1_star * v_z1z2
             - 2 * n * V_Z1Z2_star * z_1)


    dv_z1z2 = (k_2 * v_xz2
               + n * Z2_star * z_1
               + n * Z1_star * z_2
               - n * Z1_star * v_z2
               - n * V_Z2_star * z_1
               - n * Z2_star * v_z1
               - n * V_Z1_star * z_2
               - n * Z1_star * v_z1z2
               - n * V_Z1Z2_star * z_1
               - n * Z2_star * v_z1z2
               - n * V_Z1Z2_star * z_2)


    dv_z2 = (n * Z2_star * z_1
             + n * Z1_star * z_2
             - 2 * n * Z1_star * v_z2
             - 2 * n * V_Z2_star * z_1
             - 2 * n * Z2_star * v_z1z2
             - 2 * n * V_Z1Z2_star * z_2)


    return [du, dx, dz_1, dz_2,
            dv_u, dv_ux, dv_uz1, dv_uz2,
            dv_x, dv_xz1, dv_xz2,
            dv_z1, dv_z1z2, dv_z2]

def solve_and_save_new(y0_new, params_new, t_final_new=85, csv_filename=None):
    t_span = (0, t_final_new)
    t_eval = np.arange(0, t_final_new + 0.1, 0.1)
    sol = solve_ivp(
        fun=ode_system_new,
        t_span=t_span,
        y0=y0_new,


        args=(params_new['p_2'], params_new['k_1'], params_new['k_2'], params_new['k_in'], params_new['n'],
              params_new['Z1_star'], params_new['Z2_star'], params_new['X_star'],
              params_new['V_UZ1_star'], params_new['V_UZ2_star'], params_new['V_UX_star'],
              params_new['V_XZ1_star'], params_new['V_XZ2_star'], params_new['V_X_star'],
              params_new['V_Z1_star'], params_new['V_Z2_star'], params_new['V_Z1Z2_star']),
        t_eval=t_eval,
        method = 'LSODA'
    )


    df = pd.DataFrame({
        't': sol.t,

        'u': sol.y[0],
        'x': sol.y[1],
        'z_1': sol.y[2],
        'z_2': sol.y[3],

        'v_u': sol.y[4],
        'v_ux': sol.y[5],
        'v_uz1': sol.y[6],
        'v_uz2': sol.y[7],

        'v_x': sol.y[8],
        'v_xz1': sol.y[9],
        'v_xz2': sol.y[10],

        'v_z1': sol.y[11],
        'v_z1z2': sol.y[12],
        'v_z2': sol.y[13]


    })

    df.to_csv(csv_filename, index=False)
    print(f"Saved solution to '{csv_filename}'")
    return df



a = 1.15


U        = a * 0
X        = a * 20
Z_1      = a * 7.8125
Z_2      = a * 0.00256
V_U      = a * 0
V_UX     = a * 0
V_UZ1    = a * 0
V_UZ2    = a * 0
V_X      = a * 71.18322066249884
V_XZ1    = a * -19.99344557128861
V_XZ2    = a * 0.0065544287113879956
V_Z1     = a * 10.03368864557458
V_Z1Z2   = a * -0.0032870001285068215
V_Z2     = a * 0.002561077084202109


V_o = [U, X, Z_1, Z_2, V_U, V_UX, V_UZ1, V_UZ2, V_X, V_XZ1, V_XZ2, V_Z1, V_Z1Z2, V_Z2]
y0 = V_o.copy()


params = {
    'b': 250,
    'k_in': 32,
    'k_1': 1.6,
    'k_2': 1,
    'k_3': 20,
    'n': 1000,
    'c': 10,
    'p_1': 0.01,
    'p_2': 0.1
}

data_lna = solve_and_save(y0, params, t_final=85, csv_filename='lna_solution.csv')


t_grid_o = data_lna[data_lna.columns[0]]
U_f = data_lna[data_lna.columns[1]]
X_f = data_lna[data_lna.columns[2]]
Z_1_f = data_lna[data_lna.columns[3]]
Z_2_f = data_lna[data_lna.columns[4]]
V_U_f = data_lna[data_lna.columns[5]]
V_UX_f = data_lna[data_lna.columns[6]]
V_UZ1_f = data_lna[data_lna.columns[7]]
V_UZ2_f = data_lna[data_lna.columns[8]]
V_X_f = data_lna[data_lna.columns[9]]
V_XZ1_f = data_lna[data_lna.columns[10]]
V_XZ2_f = data_lna[data_lna.columns[11]]
V_Z1_f = data_lna[data_lna.columns[12]]
V_Z1Z2_f = data_lna[data_lna.columns[13]]
V_Z2_f = data_lna[data_lna.columns[14]]


U_star      = 1
X_star      = 20.0
Z_1_star     = 8.8125
Z_2_star     = 0.0022695
V_U_star    = 5.5
V_UX_star   = 0.526761
V_UZ1_star  = 5.26625
V_UZ2_star  = -0.00135621
V_X_star    = 66.573
V_XZ1_star  = -19.9948
V_XZ2_star  = 0.00515126
V_Z1_star   = 16.1564
V_Z1Z2_star = -0.00416021
V_Z2_star   = 0.00227057



y0_new = [
    U   - U_star,
    X   - X_star,
    Z_1 - Z_1_star,
    Z_2 - Z_2_star,
    V_U    - V_U_star,
    V_UX   - V_UX_star,
    V_UZ1  - V_UZ1_star,
    V_UZ2  - V_UZ2_star,
    V_X    - V_X_star,
    V_XZ1  - V_XZ1_star,
    V_XZ2  - V_XZ2_star,
    V_Z1   - V_Z1_star,
    V_Z1Z2 - V_Z1Z2_star,
    V_Z2   - V_Z2_star,
]

params_new = {
    'k_1': 1.6,
    'k_2': 1,
    'n': 1000,
    'k_in': 32,
    'p_2': 0.1,


    'Z1_star': Z_1_star,
    'Z2_star': Z_2_star,
    'X_star': X_star,

    'V_UZ1_star': V_UZ1_star,
    'V_UZ2_star': V_UZ2_star,
    'V_UX_star': V_UX_star,

    'V_XZ1_star': V_XZ1_star,
    'V_XZ2_star': V_XZ2_star,
    'V_X_star': V_X_star,

    'V_Z1_star': V_Z1_star,
    'V_Z2_star': V_Z2_star,
    'V_Z1Z2_star': V_Z1Z2_star,


}

data_lna_new = solve_and_save_new(y0_new, params_new, t_final_new=85, csv_filename='lna_solution_lin.csv')


t_grid_l = data_lna_new[data_lna_new.columns[0]]
u_f = data_lna_new[data_lna_new.columns[1]]
x_f = data_lna_new[data_lna_new.columns[2]]
z_1_f = data_lna_new[data_lna_new.columns[3]]
z_2_f = data_lna_new[data_lna_new.columns[4]]
v_u_f = data_lna_new[data_lna_new.columns[5]]
v_ux_f = data_lna_new[data_lna_new.columns[6]]
v_uz1_f = data_lna_new[data_lna_new.columns[7]]
v_uz2_f = data_lna_new[data_lna_new.columns[8]]
v_x_f = data_lna_new[data_lna_new.columns[9]]
v_xz1_f = data_lna_new[data_lna_new.columns[10]]
v_xz2_f = data_lna_new[data_lna_new.columns[11]]
v_z1_f = data_lna_new[data_lna_new.columns[12]]
v_z1z2_f = data_lna_new[data_lna_new.columns[13]]
v_z2_f = data_lna_new[data_lna_new.columns[14]]


x_output_mean = x_f + params_new['X_star']
x_output_mean = [50*value for value in x_output_mean]

x_output_var = v_x_f + params_new['V_X_star']
x_output_var = [50*value for value in x_output_var]

x_fano = [var_i/mean_i for mean_i, var_i in zip(x_output_mean,x_output_var)]



X_f_part = [50*value for value in X_f]
V_X_part = [50*value for value in V_X_f]
X_fano_part = [var_i/mean_i for mean_i, var_i in zip(X_f_part,V_X_part)]



plt.figure(figsize=(cm2inch(12), cm2inch(8)))

ax1 = plt.gca()





line1, = ax1.plot(t_grid_o, X_f_part, linestyle="solid", color="red", label = "Original [part]", alpha = 0.6)

line2, = ax1.plot(t_grid_l, x_output_mean, linestyle="solid", color="black", label = "Linearized [part]", alpha = 0.6)



ax1.set_ylabel('Mean')
plt.xlabel('Time')

plt.grid(True)
ax1.legend(handles=[line1, line2])

plt.tight_layout()

plt.savefig(folder_path_exp + "/" + "x_mean_part_paper.svg", format='svg', dpi=1000)
plt.savefig(folder_path_exp + "/" + "x_mean_part_paper.pdf", format='pdf', dpi=1000)
plt.close()


plt.figure(figsize=(cm2inch(12), cm2inch(8)))

ax1 = plt.gca()

line1, = ax1.plot(t_grid_o, X_fano_part, linestyle="solid", color="red", label = "Original [part]", alpha = 0.6)

line2, = ax1.plot(t_grid_l, x_fano, linestyle="solid", color="black", label = "Linearized [part]", alpha = 0.6)



ax1.set_ylabel('Fano')
plt.xlabel('Time')

plt.grid(True)
ax1.legend(handles=[line1, line2])

plt.tight_layout()

plt.savefig(folder_path_exp + "/" + "x_fano_part_paper.svg", format='svg', dpi=1000)
plt.savefig(folder_path_exp + "/" + "x_fano_part_paper.pdf", format='pdf', dpi=1000)
plt.close()


plt.figure(figsize=(cm2inch(12), cm2inch(8)))

ax1 = plt.gca()

line1, = ax1.plot(t_grid_o, V_X_part, linestyle="solid", color="red", label = "Original [part]", alpha = 0.6)

line2, = ax1.plot(t_grid_l, x_output_var, linestyle="solid", color="black", label = "Linearized [part]", alpha = 0.6)



ax1.set_ylabel('Variance')
plt.xlabel('Time')

plt.grid(True)
ax1.legend(handles=[line1, line2])

plt.tight_layout()

plt.savefig(folder_path_exp + "/" + "x_var_part_paper.svg", format='svg', dpi=1000)
plt.savefig(folder_path_exp + "/" + "x_var_part_paper.pdf", format='pdf', dpi=1000)
plt.close()
