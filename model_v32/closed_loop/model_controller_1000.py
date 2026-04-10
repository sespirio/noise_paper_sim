import re

def controller_model():

    model_name = "differentiator_v23.0"


    species_names = [

                     "y",


                     "x",
                     "z_1",
                     "z_2"
    ]

    species_initial_concentrations = [

                     0,


                     0,
                     0,
                     0
    ]


    parameters_names = [

                     "p_1",
                     "p_2",
                     

                     "b",
                     "k_2",
                     "k_1",
                     "k_3",
                     "n",
                     "k_in",
                     "k_f"
    ]


    parameters_initial_values = [

        1,
        1,


        250,
        1,
        16,
        20,
        1000,
        1.6,
        0.5
                        ]


    c = 1


    reactions_names = [

                     "R_1",
                     "R_2",


                     "R_3",
                     "R_4",
                     "R_5",
                     "R_6",
                     "R_7",
                     "R_8",
                     "R_9"
                       ]

    reactions_schemes = [
                     f" -> {c} y",
                     "y -> ",
                     
                     "-> x",
                     "x -> x + z_1",
                     "z_1 + x -> z_1",
                     " -> z_2",
                     "z_1 + z_2 -> ",
                     "y + x -> y + x + x",
                     "x -> x + y"
                       ]


    functions_names = ["R_1_function",
                       "R_2_function",

                       "R_3_function",
                       "R_4_function",
                       "R_5_function",
                       "R_6_function",
                       "R_7_function",
                       "R_8_function",
                       "R_9_function"
                       ]

    functions_expressions = [
                     "p_1",
                     "p_2*y",
                     "b",
                     "k_2*x",
                     "k_1*z_1*x",
                     "k_3",
                     "n*z_1*z_2",
                     "k_in*x*y",
                     "k_f*x"
                             ]

    functions_types = ["irreversible",
                       "irreversible",
                       "irreversible",
                       "irreversible",
                       "irreversible",
                       "irreversible",
                       "irreversible",
                       "irreversible",
                       "irreversible"
                       ]


    functions_mapping = []


    for expression in functions_expressions:
        mapping = {}


        for param in parameters_names:
            pattern = rf'(?<![\w]){re.escape(param)}(?![\w])'
            if re.search(pattern, expression):
                mapping[param] = "parameter"


        for species in species_names:
            pattern = rf'(?<![\w]){re.escape(species)}(?![\w])'
            if re.search(pattern, expression):
                mapping[species] = "modifier"


        functions_mapping.append(mapping)


    reactions_mapping = []


    for mapping in functions_mapping:
        reactions_mapping.append({key: key for key in mapping.keys()})


    compartment_volume_variable = True

    ss_value = {"y":1000}


    runs_stochastic = 10*1000


    t_f = 15


    method_sim = "directMethod"


    return model_name, species_names, species_initial_concentrations, parameters_names, parameters_initial_values, \
           reactions_names, reactions_schemes, functions_names, functions_expressions, functions_types, functions_mapping, \
           reactions_mapping, runs_stochastic, t_f, method_sim, ss_value, compartment_volume_variable