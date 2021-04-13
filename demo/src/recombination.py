import numpy as np


def NSGA_II(parents, params):
    children = []

    np.random.shuffle(parents)

    i = 0
    while i < params['children_size'] - 1:
        children.append(one_point(parents[i], parents[i + 1], params))
        children.append(one_point(parents[i + 1], parents[i], params))
        i += 2

    return parents, children


def one_point(mother, father, params):
    child = {'meta': {'error': None,
                      'size': None,
                      'dominates': None,
                      'dominated': None,
                      'k_value': None},
             'data': None}

    i = np.random.randint(params['features'])

    child['data'] = mother['data'][:i] + father['data'][i:]

    if np.random.random() < 0.5:
        child['meta']['k_value'] = mother['meta']['k_value']
    else:
        child['meta']['k_value'] = father['meta']['k_value']

    return child