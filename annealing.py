import numpy as np
import matplotlib.pyplot as plt
import random
from tsp_utils import distance_matrix, get_nearest_neighbour_solution, get_rand_solution, animation_tsp_v2



class Annealing():

    
    def __init__(self, 
                 coordinates: np.ndarray, 
                 betaminmax: list, 
                 alpha: float,  
                 maxiter: int, 
                 Ns: int = 5, 
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
        initial_sol_funcs = {'rand': get_rand_solution, 'NN': get_nearest_neighbour_solution}
        if initial_sol not in initial_sol_funcs:
            raise ValueError("initial_sol must either be 'rand' or 'NN' ")
        
        
        # coordinates of the cities
        self.coords = coordinates
        # number of cities
        self.num_cities = len(self.coords)
        
        # for annealing
        self.beta, self.beta_max = betaminmax
        self.alpha = alpha
        self.maxiter = maxiter
        # number of sweeps before beta is changed
        self.Ns = Ns
        
        # get distance matrix
        self.d_ij = distance_matrix(self.coords)
 
        # based on kwarg, initial solution is either random or NN
        self.current_tour = initial_sol_funcs[initial_sol](self.d_ij)
        
        # save the shortest reached tour
        self.shortest_tour = self.current_tour
        # save tour of each iteration in history for later animation and analysis
        self.tour_history = [self.current_tour]
        
        # same for the energy
        self.current_energy = self.get_energy(self.current_tour)
        self.lowest_energy = self.current_energy
        self.energy_history = [self.current_energy]
        
    
    def get_energy(self, tour):
        """Calculates the total path length of the current tour

        Args:
            tour (list): list of city-numbers (integers) which defines the sequence of travelling

        Returns:
            H (float): The total energy (path-length) of the tour  
        """
        return np.sum([self.d_ij[i,j] for i,j in zip(tour, np.roll(tour, -1))])
    
    
    def p_accept(self, proposed_energy):
        """Evaluates the acceptance propability for accepting the proposed path

        Args:
            proposed_energy (float): path-length of proposed path

        Returns:
            float: value between 0 and 1 depending on delta_H and beta
        """
        return np.exp(-np.abs(proposed_energy - self.current_energy)*self.beta)
    
    
    def make_proposal(self):
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
            i, j = min(i,j), max(i,j)
            subroute = proposed_tour[i:j]
            del proposed_tour[i:j]
            insert_pos = np.random.choice(len(proposed_tour))
            proposed_tour[insert_pos:insert_pos] = subroute
            return proposed_tour
        
        get_new_tour = np.random.choice([_swap_cities, _relocate_path, _reverse_path])
        proposal = get_new_tour(self.current_tour)
        return proposal
    
    
    def accept_or_reject(self, proposed_tour):
        """_summary_

        Args:
            proposed_tour (_type_): _description_
        """
        
        proposal_energy = self.get_energy(proposed_tour)
        if proposal_energy < self.current_energy:
            self.current_energy = proposal_energy
            self.current_tour = proposed_tour
            if self.current_energy < self.lowest_energy:
                self.lowest_energy = self.current_energy
                self.shortest_tour = self.current_tour
                
        else:
            if random.random() < self.p_accept(proposal_energy):
                self.current_energy = proposal_energy
                self.current_tour = proposed_tour


    def anneal(self):
        
        iteration = 0
        while self.beta <= self.beta_max and iteration < self.maxiter:
            for _ in range(self.Ns):
        
                proposed_tour = self.make_proposal()
                self.accept_or_reject(proposed_tour)
                # update histories
                self.energy_history.append(self.current_energy)
                self.tour_history.append(self.current_tour)
                iteration += 1
                
            self.beta *= self.alpha
            
        return
    
    
    def plot_energy_hist(self):
        
        plt.figure()
        plt.title('TSP Solution')
        plt.xlabel('iterations')
        plt.ylabel('energy')
        plt.plot([i for i in range(len(self.energy_history))], self.energy_history)
        plt.show()
        return
    
    

    
    
    def animate_solution(self):
        animation_tsp_v2(self.tour_history, self.energy_history, self.coords)
        return
        
        

    
    
