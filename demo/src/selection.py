import numpy as np
import math


def NSGA_II(population, params, result_size):
    selection = []
    j, k = 0, 0

    fronts = fast_nondominated_sort(population, params)

    while len(selection) + len(fronts[j]) < result_size:
        for individual in fronts[j]:
            selection.append(individual)
        j += 1

    np.random.shuffle(fronts[j])

    while len(selection) < result_size:
        selection.append(fronts[j][k])
        k += 1

    return selection, fronts


def fast_nondominated_sort(population, params):
    fronts, k = np.empty(params['population_size'], dtype=set), 0

    for i in range(params['population_size']):
        fronts[i] = []

    for i in range(len(population)):
        population[i]['meta']['dominates'], population[i]['meta']['dominated'] = set(), 0
        for j in range(len(population)):
            if population[i]['meta']['error'] < population[j]['meta']['error'] and population[i]['meta']['size'] <= population[j]['meta']['size']:
                population[i]['meta']['dominates'].add(j)
            if population[j]['meta']['error'] < population[i]['meta']['error'] and population[j]['meta']['size'] <= population[i]['meta']['size']:
                population[i]['meta']['dominated'] += 1
        if population[i]['meta']['dominated'] == 0:
            fronts[0].append(population[i])

    while len(fronts[k]) > 0:
        for i in range(len(fronts[k])):
            for j in fronts[k][i]['meta']['dominates']:
                population[j]['meta']['dominated'] -= 1
                if population[j]['meta']['dominated'] == 0:
                    fronts[k + 1].append(population[j])
        k += 1

    return fronts


def crowding_distance_assignment(front):
    objectives = ["fitness", "novelty"]
    for objective in objectives:
        front = sorted(front, key=lambda individual: individual["meta"][objective])
        front[0]["meta"]["distance"], front[-1]["meta"]["distance"] = math.inf, math.inf
        for i in range(2, len(front) - 1):
            front[i]["meta"]["distance"] += front[i + 1]["meta"][objective] - front[i - 1]["meta"][objective]
    return front
