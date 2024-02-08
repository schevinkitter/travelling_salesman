import numpy as np
from annealing import Annealing
from tsp_utils import generate_rand_cities


def main():

    coords = generate_rand_cities(10)

    tsm1 = Annealing(coordinates=coords,
                     betaminmax=[0.001, 50],
                     alpha=1.1,
                     maxiter=10_000)
    
    tsm1.anneal()
    
    tsm1.plot_energy_hist()
    
    tsm1.animate_solution()




if __name__ == '__main__':
    main()