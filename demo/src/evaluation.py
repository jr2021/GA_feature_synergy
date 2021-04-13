import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
import math


def NSGA_II(population, data, params):
    for i in range(len(population)):
        if population[i]['meta']['error'] is None:
            population[i]['meta']['size'] = sum(population[i]['data'])
            if population[i]['meta']['size'] > 0:
                population[i]['meta']['error'] = KNN_KFold(population[i], data.copy(), params)
            else:
                population[i]['meta']['error'] = 1
                population[i]['size']['size'] = params['features']


    return population


def KNN_KFold(individual, data, params):
    model, cross_validation = KNeighborsClassifier(individual['meta']['k_value']), StratifiedKFold(5, shuffle=True, random_state=1)
    scores = []

    X, y = data.drop(columns=params['y']), data[params['y']]

    x = X.copy()
    for i in range(len(X.columns[:])):
        if individual['data'][i] == 0:
            x = x.drop(columns=X.columns[i])
        else:
            x[X.columns[i]] *= individual['data'][i]

    for i, j in cross_validation.split(x, y):
        x_train, x_test, y_train, y_test = x.iloc[i], x.iloc[j], y.iloc[i], y.iloc[j]
        model.fit(x_train, y_train)
        scores.append(model.score(x_test, y_test))

    return 1 - np.array(scores).mean()
