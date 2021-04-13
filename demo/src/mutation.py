import numpy as np


def flip(children, params):
    for i in range(params['children_size']):
        j = np.random.randint(params['features'])
        if children[i]['data'][j] == 1:
            if sum(children[i]['data']) <= 1:
                pass
            else:
                children[i]['data'][j] = 0
        else:
            children[i]['data'][j] = 1
        if children[i]['meta']['k_value'] <= 1:
            children[i]['meta']['k_value'] += np.random.choice([0, 1])
        else:
            children[i]['meta']['k_value'] += np.random.choice([-1, 0, 1])
    
    return children

