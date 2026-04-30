
import sys
if '../..' not in sys.path:
    sys.path.append('../..')

from basico import *
from joblib import dump

from model_controller_1000 import controller_model



model_name,species_names,species_initial_concentrations,parameters_names,parameters_initial_values,\
           reactions_names,reactions_schemes,functions_names,functions_expressions,functions_types,functions_mapping, \
           reactions_mapping,runs_stochastic,t_f,method_sim, ss_value, compartment_volume_variable \
            = controller_model()


current_dir = os.getcwd()
folder_name = "stochastic_simulation_" + model_name
folder_path = os.path.join(current_dir, folder_name)
if not os.path.exists(folder_path):
    os.makedirs(folder_path)


data = {
    "Model Name": [model_name],
    "Species Names": [species_names],
    "Species Initial Concentrations": [species_initial_concentrations],
    "Parameters Names": [parameters_names],
    "Parameters Initial Values": [parameters_initial_values],
    "Reactions Names": [reactions_names],
    "Reactions Schemes": [reactions_schemes],
    "Functions Names": [functions_names],
    "Functions Expressions": [functions_expressions],
    "Functions Types": [functions_types],
    "Functions Mapping": [functions_mapping],
    "Reactions Mapping": [reactions_mapping],
    "Runs Stochastic": [runs_stochastic],
    "Final Time": [t_f],
    "Simulation Method": [method_sim],
    "Steady State Value": [ss_value],
    "Compartment Volume Variable": [compartment_volume_variable]
}


df = pd.DataFrame(data)


csv_file_path = folder_path + "/" + "model_details.csv"
df.to_csv(csv_file_path, index=False)


new_model(name=model_name)


for i, species_i in enumerate(species_names):
    add_species(name = species_i, initial_concentration = species_initial_concentrations[i])


for i, parameter_i in enumerate(parameters_names):
    add_parameter(name = parameter_i, initial_value = parameters_initial_values[i])


for i, function_i in enumerate(functions_names):
    add_function(name = function_i, infix = functions_expressions[i], type = functions_types[i],
                 mapping=functions_mapping[i])


for i, reaction_i in enumerate(reactions_names):
    add_reaction(name = reaction_i, scheme = reactions_schemes[i], function = functions_names[i],
                 mapping = reactions_mapping[i])


run_steadystate(update_model=False)

if compartment_volume_variable == True:
    c_ss = get_species(list(ss_value.keys())[0])[['concentration']].values
    c_ss = c_ss.tolist()[0][0]
    avogrado_c = 6.02214076e23
    set_ss_particle = list(ss_value.values())[0]
    compartment_volume = set_ss_particle/(c_ss*avogrado_c)

    set_compartment(name="compartment", initial_size=compartment_volume)


    for i, species_i in enumerate(species_names):
        set_species(name=species_i, initial_concentration=species_initial_concentrations[i])

else:
    print("compartment_volume_variable == False. No compartment volume defined.")
    sys.exit()




add_event(name=f'parm_k_in', trigger='{Time}=={15}',
          assignments=[[f'Values[k_in]', '{Values[k_in].InitialValue}+32']])

add_event(name=f'parm_p_1', trigger='{Time}=={15}',
          assignments=[[f'Values[p_1]', '{Values[p_1].InitialValue}+0.01']])

add_event(name=f'parm_p_2', trigger='{Time}=={15}',
          assignments=[[f'Values[p_2]', '{Values[p_2].InitialValue}+0.1']])

save_model(folder_path + "/" + model_name + '.cps')


print(get_reactions()[['scheme']])
print(get_reactions()[['function']])
print(get_reactions()[['mapping']])
print(get_parameters()[['initial_value']])


result_det = run_time_course(duration=t_f, automatic=False, method="deterministic", max_steps=1000000, stepsize=0.1,
                         use_numbers=True)


t_index_det = result_det[species_names[0]].keys()
t_index_det = t_index_det.tolist()

dict_of_states_det = {key_a: {key_b: [] for key_b in t_index_det} for key_a in species_names}

for key_a in species_names:
    for key_b in t_index_det:
        dict_of_states_det[key_a][key_b].append(result_det[key_a][key_b])


dump(dict_of_states_det, folder_path + "/dict_of_states_det.joblib")
dump(t_index_det, folder_path + "/t_index_det.joblib")


count_save = 100
count_it = 0

for k in range(runs_stochastic):
    result = run_time_course(duration = t_f, automatic = False, method = method_sim, max_steps = 1000000,
                             stepsize= 0.1, use_numbers=True)

    print("Monte Carlo simulation, iteration: ", k)


    t_index = result[species_names[0]].keys()
    t_index = t_index.tolist()

    if k == 0:

        dict_of_states = {key_a: {key_b: [] for key_b in t_index} for key_a in species_names}
    else:
        pass

    for key_a in species_names:
        for key_b in t_index:
            dict_of_states[key_a][key_b].append(result[key_a][key_b])

    count_it += 1


    if count_it == count_save:

        dump(dict_of_states, folder_path + "/dict_of_states_" + str(k) + ".joblib", compress=3)
        dump(t_index, folder_path + "/t_index.joblib")
        print("Batch of data saved.")


        dict_of_states = {key_a: {key_b: [] for key_b in t_index} for key_a in species_names}
        count_it = 0


    if k == list(range(runs_stochastic))[-1] and count_it != 0:
        dump(dict_of_states, folder_path + "/dict_of_states_" + str(k) + ".joblib", compress=3)
        dump(t_index, folder_path + "/t_index.joblib")
        print("Batch of data saved.")

