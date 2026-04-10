import re

def controller_model():

    model_name = "differentiator_v25"


    species_names = [
                     "x",
                     "z_1",
                     "z_2"
                     ]

    species_initial_concentrations = [
                     0,
                     0,
                     0
    ]


    parameters_names = [
                     "b",
                     "k_2",
                     "k_1",
                     "k_3",
                     "n",

                     "d"
                        ]


    parameters_initial_values = [
                     250,
                     1,
                     1.6,
                     20,
                     1000,

                     0.1
                        ]


    reactions_names = [
                     "R_1",
                     "R_2",
                     "R_3",
                     "R_4",
                     "R_5",

                     "R6"
                       ]

    reactions_schemes = [
                     "-> x",
                     "x -> x + z_1",
                     "z_1 + x -> z_1",
                     " -> z_2",
                     "z_1 + z_2 -> ",

                     "x -> "
                       ]


    functions_names = ["R_1_function",
                       "R_2_function",
                       "R_3_function",
                       "R_4_function",
                       "R_5_function",

                       "R_6_function"
                       ]

    functions_expressions = ["b",
                             "k_2*x",
                             "k_1*z_1*x",
                             "k_3",
                             "n*z_1*z_2",

                             "d*x"
                             ]

    functions_types = ["irreversible",
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

    ss_value = {"x":1000}


    runs_stochastic = 10*1000


    t_f = 15


    method_sim = "directMethod"


    return model_name, species_names, species_initial_concentrations, parameters_names, parameters_initial_values, \
           reactions_names, reactions_schemes, functions_names, functions_expressions, functions_types, functions_mapping, \
           reactions_mapping, runs_stochastic, t_f, method_sim, ss_value, compartment_volume_variable