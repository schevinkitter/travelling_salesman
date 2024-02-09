import numpy as np
from annealing import Annealing
from tsp_utils import generate_rand_cities, load_tsp_problem


def main():

    coords = generate_rand_cities(20)
    # coords = load_tsp_problem('data/pr124.tsp')

    # tsm1 = Annealing(coordinates=coords,
    #                  betaminmax=[0.001, 400000],
    #                  alpha=1.001,
    #                  maxiter=100_000)

    # tsm1.repeat_annealing(3)
    # # tsm1.plot_energy_hist()
    # tsm1.animate_solution()

    tsm2 = Annealing(coordinates=coords,
                     betaminmax=[1, 150],
                     alpha=0.1,    #1.01
                     maxiter=100_000)

    tsm2.repeat_annealing(5)
    tsm2.animate_solution()

    # tsm2.plot_energy_hist()
    tsm2.show_result()
    tsm2.print_results()


if __name__ == '__main__':
    main()
