import numpy as np
import matplotlib.pyplot as plt
import random
from tsp_utils import distance_matrix, get_nearest_neighbour_solution, get_rand_solution, animation_tsp, show_endresult


class Annealing():

    def __init__(self,
                 coordinates: np.ndarray,
                 betaminmax: list,
                 alpha: float,
                 maxiter: int,
                 Ns: int = 0,
                 initial_sol: str = 'rand') -> None:
        """Class for perfroming simulated annealing for solving the travelling salesman problem

        Args:
            coordinates (np.ndarray): the columns contain x and y coordinates of the cities
            betaminmax (list): min and max inverse temperature for annealing
            alpha (float): factor by which beta gets increased after Ns iterations
            maxiter (int): maximum number of interations
            Ns (int, optional): number of iterations for which beta stays constant. Defaults to 5.
            initial_sol (str, optional): The default method for numbering the initial solution. Currently
                                         implemented are random ('rand') and nearest neighbours ('NN') 
                                         Defaults to 'rand'.

        """

        # initial solutions
        initial_sol_funcs = {'rand': get_rand_solution,
                             'NN': get_nearest_neighbour_solution}
        if initial_sol not in initial_sol_funcs:
            raise ValueError("initial_sol must either be 'rand' or 'NN' ")

        # coordinates of the cities
        self.coords = coordinates
        # number of cities
        self.num_cities = len(self.coords)

        # for annealing
        self.beta_init = betaminmax[0]
        self.beta, self.beta_max = betaminmax
        self.alpha = alpha
        self.maxiter = maxiter
        self.iter_counter = 0
        # number of sweeps before beta is changed
        if Ns == 0:
            self.Ns = self.num_cities //3
        else: 
            self.Ns = Ns

        # get distance matrix
        self.d_ij = distance_matrix(self.coords)

        # based on kwarg, initial solution is either random or NN
        self.current_tour = initial_sol_funcs[initial_sol](self.d_ij)

        # save the shortest reached tour
        self.shortest_tour = self.current_tour
        # save tour of each iteration in history for later animation and analysis
        self.total_tour_history = []

        # same for the energy
        self.current_energy = self.get_energy(self.current_tour)
        self.lowest_energy = self.current_energy
        self.total_energy_history = []

        # if number of rejections reaches a threshold of 1000, algorithm stopps
        self.rejection_counter = 0
        self.idx_of_lowest_energy = None



    def get_energy(self, tour):
        """Calculates the total path length of the current tour

        Args:
            tour (list): list of city-numbers (integers) which defines the sequence of travelling

        Returns:
            H (float): The total energy (path-length) of the tour  
        """
        return np.sum([self.d_ij[i, j] for i, j in zip(tour, np.roll(tour, -1))])

    def _get_p_accept(self, proposed_energy):
        """Evaluates the acceptance propability for accepting the proposed path

        Args:
            proposed_energy (float): path-length of proposed path

        Returns:
            float: value between 0 and 1 depending on delta_H and beta
        """
        return np.exp(-np.abs(proposed_energy - self.current_energy)*self.beta)

    def _make_proposal(self):
        """Chooses randomly between the three propsal functions and creates a new
        proposal by calling them. 
        """

        def _swap_cities(current_tour):
            """swap two random cities
            """
            proposed_tour = current_tour.copy()
            i, j = np.random.choice(proposed_tour, size=2, replace=False)
            proposed_tour[i], proposed_tour[j] = proposed_tour[j], proposed_tour[i]
            return proposed_tour

        def _reverse_path(current_tour):
            """choose a random path and reverse order of visiting
            """
            proposed_tour = current_tour.copy()
            i, j = np.random.choice(proposed_tour, size=2, replace=False)
            i, j = min(i, j), max(i, j)
            proposed_tour[i:j] = reversed(proposed_tour[i:j])

            return proposed_tour

        def _relocate_path(current_tour):
            """choose random path and insert it at random position
            """
            proposed_tour = current_tour.copy()
            i, j = np.random.choice(proposed_tour, size=2, replace=False)
            i, j = min(i, j), max(i, j)
            subroute = proposed_tour[i:j]
            del proposed_tour[i:j]
            insert_pos = np.random.choice(len(proposed_tour))
            proposed_tour[insert_pos:insert_pos] = subroute
            return proposed_tour

        get_new_tour = np.random.choice(
            [_swap_cities, _relocate_path, _reverse_path])
        proposal = get_new_tour(self.current_tour)
        return proposal

    def _accept_or_reject(self, proposed_tour):
        """_summary_

        Args:
            proposed_tour (_type_): _description_
        """

        proposal_energy = self.get_energy(proposed_tour)
        if proposal_energy < self.current_energy:
            self.rejection_counter = 0
            self.current_energy = proposal_energy
            self.current_tour = proposed_tour
            if self.current_energy < self.lowest_energy:
                self.lowest_energy = self.current_energy
                self.shortest_tour = self.current_tour
                self.idx_of_lowest_energy = self.iter_counter


        elif random.random() < self._get_p_accept(proposal_energy):
            self.rejection_counter = 0
            self.current_energy = proposal_energy
            self.current_tour = proposed_tour


        else:
            self.rejection_counter += 1
            


    def _perform_annealing_sweep(self):
        """Performs one sweep of annealing and records energy and tour histories
        """
        # reset annealing parameters sweep histories and iteration counter
        self._reset_parameters()
    
        while self.beta <= self.beta_max and self.iter_counter < self.maxiter and self.rejection_counter < 1_000:
            for _ in range(self.Ns):
                
                proposed_tour = self._make_proposal()
                self._accept_or_reject(proposed_tour)
                # update histories
                self.energy_history_sweep.append(self.current_energy)
                self.tour_history_sweep.append(self.current_tour)
                self.iter_counter += 1
                
            self.beta += self.alpha
        return
    
    def _reset_parameters(self):
        """Resets the annealing parameters and histories for a new
        annealing sweep
        """
        # reset inverse temperature
        self.beta = self.beta_init
        self.iter_counter = 0
        
        # start from the shortes tour found yet
        # TODO: maybe just start from the last position
        # self.current_tour = self.shortest_tour
        # self.current_energy = self.lowest_energy
        self.idx_of_lowest_energy = 0
        
        # reset sweep histories
        self.tour_history_sweep = [self.current_tour]
        self.energy_history_sweep = [self.current_energy]
        return


    def repeat_annealing(self, M: int):
        """perfroms M annealing sweeps and records data

        Args:
            M (int): number of annealing sweeps
        """
        self.shortest_tours = []
        self.lowest_energies = []
        self.iterations_sweeps = []
        
        for sweep in range(M):
            self._perform_annealing_sweep()
            
            lowest_energy = self.energy_history_sweep[self.idx_of_lowest_energy]
            shortest_tour = self.tour_history_sweep[self.idx_of_lowest_energy]
            
            self.lowest_energies.append(lowest_energy)
            self.shortest_tours.append(shortest_tour)
            # not really needed
            self.iterations_sweeps.append(self.iter_counter)
            # only needed for simulation
            self.total_tour_history.extend(self.tour_history_sweep)
            # needed for plot of energy
            self.total_energy_history.extend(self.energy_history_sweep)
        
        return
    

    def plot_energy_hist(self):
        fig, ax = plt.subplots()
        ax.plot([i for i in range(len(self.total_energy_history))], self.total_energy_history)
        ax.set_xlim(0, len(self.total_energy_history))
        ax.set_ylim(0, max(self.total_energy_history) + 0.1*max(self.total_energy_history))
        ax.set_ylabel(r'$H$')
        ax.set_xlabel(r'iterations')
        # Setting xticks
        # Getting automatic ticks
        xticks = list(ax.get_xticks())
        xticks.pop()
        # Adding one single tick to the automatic ticks
        xticks[-1] = len(self.total_energy_history)
        # Setting xticks
        ax.set_xticks(xticks)
        ax.axhline(y = min(self.total_energy_history), color='green', linestyle='--', label = r'$H_{min}$')
        plt.show()
        return

    def animate_solution(self):
        animation_tsp(self.total_tour_history, self.total_energy_history, self.coords)
        return

    def show_result(self):
        show_endresult(self.shortest_tour, self.total_energy_history, self.coords)
        return
    
    def print_results(self):
        
        """Prints out the lowest energies of all sweeps."""
        print("Lowest energies of all sweeps:")
        for i, energy in enumerate(self.lowest_energies):
            print(f"Sweep {i + 1}: {energy}")
        
        return
    
    
