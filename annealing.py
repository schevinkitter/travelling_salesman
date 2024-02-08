import numpy as np
import matplotlib.pyplot as plt
import random
from tsp_utils import distance_matrix, get_nearest_neighbour_solution
from tsp_animation import animation_tsp


class Annealing():
    
    def __init__(self, coordinates: np.ndarray, betaminmax: list, alpha: float,  maxiter: int, Ns: int = 5 ) -> None:
        
        # coordinates of the cities
        self.coords = coordinates
        # number of cities
        self.num_cities = len(self.coords)
        
        # for annealing
        # start with beta_min and increase with factor alpha until beta_max is reached
        self.beta, self.beta_max = betaminmax
        self.alpha = alpha
        self.maxiter = maxiter
        # number of sweeps before beta is changed
        self.Ns = Ns
        
        # get distance matrix and NN-solution as initial solution
        self.d_ij = distance_matrix(self.coords)
        # tour in each iteration
        self.current_tour = get_nearest_neighbour_solution(self.d_ij)
        # save the shortest reached tour
        self.shortest_tour = self.current_tour
        # save tour of each iteration in history for later animation and analysis
        self.tour_history = [self.current_tour]
        
    
        # save energy history 
        self.current_energy = self.get_energy(self.current_tour)
        self.lowest_energy = self.current_energy
        self.energy_history = [self.current_energy]
        
    
    def get_energy(self, tour):
       
        return np.sum([self.d_ij[i,j] for i,j in zip(tour, np.roll(tour, -1))])
    
    
    def p_accept(self, proposed_energy):
        
        return np.exp(-np.abs(proposed_energy - self.current_energy)*self.beta)
    
    
    def make_proposal(self):
        
        
        def _swap_cities(current_tour):
            """Update proposal 1
            """
            proposed_tour = current_tour.copy()
            i, j = np.random.choice(proposed_tour, size=2, replace=False)
            proposed_tour[i], proposed_tour[j] = proposed_tour[j], proposed_tour[i]
            return proposed_tour
        
        
        def _reverse_path(current_tour):
            """Update proposal 2
            """
            proposed_tour = current_tour.copy()

            i, j = np.random.choice(proposed_tour, size=2, replace=False)
            i, j = min(i, j), max(i, j)
            
            proposed_tour[i:j] = reversed(proposed_tour[i:j])

            return proposed_tour

        def _relocate_path(current_tour):
            """Update proposal 3
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
            for sweep in range(self.Ns):
        
                proposed_tour = self.make_proposal()
                self.accept_or_reject(proposed_tour)
                # history stuff here
                
                
                iteration += 1
                self.energy_history.append(self.current_energy)
                self.tour_history.append(self.current_tour)
                
                
            self.beta *= self.alpha
            
        return
    
    def plot_energy_hist(self):
        fig = plt.figure()
        plt.plot([i for i in range(len(self.energy_history))], self.energy_history)
        plt.show()
        
        return
    
    def animate_solution(self):
        
        animation_tsp(self.tour_history, self.coords)
        
        return
        
        

    
    
