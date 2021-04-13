import pandas as pd
import numpy as np
import initialisation as init
import selection as sel
import recombination as rec
import mutation as mut
import evaluation as eva
from statistics import Statistics
import time

def evolve():
    params = {'population_size': 200,
              'parents_size': 100,  # must be even
              'children_size': 100,
              'generations': 1000,
              'mutation_rate': 1,
              'k_range': 30,
              'features': 163,
              'data_file': 'OA_processed.csv',
              'y': 'OA'}

    data = pd.read_csv(params['data_file'])

    instance = Statistics(params)

    population = init.NSGA_II(params)

    population = eva.NSGA_II(population, data, params)

    for generation in range(params['generations']):

        parents, fronts = sel.NSGA_II(population, params)

        parents, children = rec.NSGA_II(parents, params)

        children = mut.flip(children, params)

        children = eva.NSGA_II(children, data, params)

        population = np.concatenate((parents, children))

        instance.update_dynamic(population, generation)

    instance.update_static(population)

    instance.print_json()


def main():
    evolve()


main()
