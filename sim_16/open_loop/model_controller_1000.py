import re

def controller_model():

    model_name = "differentiator_v31.0"


    species_names = [
                     
                     "y"

    ]

    species_initial_concentrations = [

                     0 

    ]


    parameters_names = [

                     "p_1", 
                     "p_2"  

    ]


    parameters_initial_values = [

        1, 
        1 

                        ]


    c = 1


    reactions_names = [
                    
                     "R_1",
                     "R_2"

                       ]

    reactions_schemes = [
                     f" -> {c} y", 
                     "y -> " 

                       ]


    functions_names = ["R_1_function",
                       "R_2_function"

                       ]

    functions_expressions = [
                     "p_1",1
                     "p_2*y" 

                             ]

    functions_types = ["irreversible",  
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


    compartment_volume_variable = False

    ss_value = {"y":1000}


    runs_stochastic = 10*1000


    t_f = 15


    method_sim = "directMethod"


    return model_name, species_names, species_initial_concentrations, parameters_names, parameters_initial_values, \
           reactions_names, reactions_schemes, functions_names, functions_expressions, functions_types, functions_mapping, \
           reactions_mapping, runs_stochastic, t_f, method_sim, ss_value, compartment_volume_variable