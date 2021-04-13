import numpy as np


def NSGA_II(params):
    population = []
    for i in range(params['population_size']):
        population.append({'meta': {'error': None,
                                    'size': None,
                                    'dominates': set(),
                                    'dominated': 0,
                                    'k_value': np.random.randint(1, params['k_range'])},
                           'data': np.random.randint(2, size=params['features']).tolist()})
    return population
