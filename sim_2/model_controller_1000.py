import re


def controller_model():

    model_name = "input_network"


    species_names = [

        "u"
    ]

    species_initial_concentrations = [

        0
    ]


    parameters_names = [

        "p_1",
        "p_2"

    ]


    parameters_initial_values = [
        0.01,
        0.1
    ]


    c = 10


    reactions_names = [
        "R_7",
        "R_8"
    ]

    reactions_schemes = [
        f" -> {c} u",
        "u -> ",
    ]


    functions_names = [
                       "R_7_function",
                       "R_8_function"
                       ]

    functions_expressions = [

                             "p_1",
                             "p_2*u"
                             ]

    functions_types = [
                       "irreversible",
                       "irreversible",
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


    compartment_volume_variable = False

    ss_value = {"x": None}


    runs_stochastic = 10 * 1000


    t_f = 85


    method_sim = "directMethod"

    return model_name, species_names, species_initial_concentrations, parameters_names, parameters_initial_values, \
        reactions_names, reactions_schemes, functions_names, functions_expressions, functions_types, functions_mapping, \
        reactions_mapping, runs_stochastic, t_f, method_sim, ss_value, compartment_volume_variable