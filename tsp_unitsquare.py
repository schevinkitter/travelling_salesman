import matplotlib.pyplot as plt
import numpy as np
from numpy import random
import scipy
 
 
class TSP():

    def __init__(self, N, data_path=None, beta_min=4, beta_max=100, M=10000, Ns=30):
        # number of cities

        self.beta = beta_min
        self.beta_max = beta_max
        self.M = M
        self.Ns = Ns

        if not data_path:
            self.N = N
            self.generate_rand_cities()

        else:
            self.tsp_path = data_path
            self.load_tsp_problem()

        # three proposal methods
        self.proposals = (self._swap_cities,
                          self._reinsert_swapped_routes,
                          self._reinsert_routes)

        pass

    def generate_rand_cities(self):
        # for a first configuration, the cities are sorted by increasing x-value
        # x_coords = np.sort(random.rand(self.N))
        x_coords = random.rand(self.N)
        y_coords = random.rand(self.N)
        initial_tour = np.array(
            [[int(i), x_coords[i], y_coords[i]] for i in range(self.N)])
        # calculte distance matrix between each pair of cities
        self.dmatrix = scipy.spatial.distance_matrix(
            initial_tour[:, 1:], initial_tour[:, 1:])
        self.current_tour = initial_tour
        return

    def load_tsp_problem(self):
        import tsplib95
        data = tsplib95.load(self.tsp_path)

        node_coords = data.node_coords

        initial_tour = np.array([[int(city-1), coords[0], coords[1]]
                                for city, coords in node_coords.items()])

        self.dmatrix = scipy.spatial.distance_matrix(
            initial_tour[:, 1:], initial_tour[:, 1:])
        self.current_tour = initial_tour
        self.N = len(self.current_tour)
        return

    @staticmethod
    def compute_energy(tour):
        H = 0
        distances = scipy.spatial.distance_matrix(
            tour[:, 1:], tour[:, 1:])
        cities = tour[:, 0].astype(int)
        for i in range(len(cities)):
            from_city = i
            to_city = (i+1) % len(cities)
            H += distances[from_city, to_city]
        return H

    @staticmethod
    def compute_energy_alternative(tour):
        H = 0
        cities, x, y = tour.T
        for i in range(len(tour)):
            dx = x[(i+1) % len(cities)] - x[i]
            dy = y[(i+1) % len(cities)] - y[i]

            H += np.sqrt(dx**2 + dy**2)
        return H

    def _compute_energy(self, tour):
        H = 0
        cities, x, y = tour.T
        for i in range(len(tour)):
            dx = x[(i+1) % len(cities)] - x[i]
            dy = y[(i+1) % len(cities)] - y[i]

            H += np.sqrt(dx**2 + dy**2)
        return H

    def _swap_cities(self):
        """Update proposal 1
        """
        proposed_tour = self.current_tour
        cities = proposed_tour[:, 0].astype(int)
        i, j = random.choice(cities, size=2, replace=False)
        i, j = int(i), int(j)
        proposed_tour[[i, j]] = proposed_tour[[j, i]]

        return proposed_tour

    def _reinsert_swapped_routes(self):
        """Update proposal 2, checked
        """
        proposed_tour = self.current_tour
        cities = proposed_tour[:, 0].astype(int)
        i, j = random.choice(cities, size=2, replace=False)
        subroute = proposed_tour[min(i, j):max(i, j)]
        proposed_tour = np.delete(
            proposed_tour, slice(min(i, j), max(i, j)), axis=0)
        insert_pos = random.choice(len(proposed_tour))
        proposed_tour = np.insert(
            proposed_tour, insert_pos, subroute[::-1], axis=0)

        return proposed_tour

    def _reinsert_routes(self):
        """Update proposal 3
        """
        proposed_tour = self.current_tour
        cities = proposed_tour[:, 0].astype(int)
        i, j = random.choice(cities, size=2, replace=False)

        subroute = proposed_tour[min(i, j):max(i, j)]
        proposed_tour = np.delete(
            proposed_tour, slice(min(i, j), max(i, j)), axis=0)
        insert_pos = random.choice(len(proposed_tour))
        # proposed_tour[insert_pos] = subroute
        proposed_tour = np.insert(proposed_tour, insert_pos, subroute, axis=0)
        return proposed_tour

    def anneal(self):
        alpha = 1.1
        H_list = []
        self.H_current = self._compute_energy(self.current_tour)
        for i in range(self.M):
            # do Ns sweeps before lowering temperature
            for j in range(self.Ns):

                H_list.append(self.H_current)

                # make proposal
                proposed_tour = random.choice(self.proposals)()
                # proposed_tour = self._swap_cities()

                H_prop = self._compute_energy(proposed_tour)

                delta_H = H_prop - self.H_current

                if delta_H < 0:
                    # print('found better way')
                    self.current_tour = proposed_tour
                    self.H_current = H_prop

                else:
                    p_accept = min(1, np.exp(-delta_H*self.beta))
                    # print(np.exp(-self.beta*delta_H))
                    if random.rand() < p_accept:
                        # print(f'accepted with {p_accept}')
                        self.current_tour = proposed_tour
                        self.H_current = H_prop

            self.beta *= alpha
            if self.beta > self.beta_max:
                break

        return H_list


def plot_tour(tour, title: str = 'TSP-Solution'):
    _, x, y = tour.T

    cities = _.astype(int)
    # Plot the tour (connecting lines between cities)
    fig = plt.figure()
    for i in range(len(tour) - 1):

        from_city = cities[i]
        to_city = cities[i + 1]
        plt.plot([x[from_city], x[to_city]],
                 [y[from_city], y[to_city]], 'b-', marker='o')

    # Connect the last city to the starting city to complete the tour
    plt.plot([x[cities[-1]], x[cities[0]]],
             [y[cities[-1]], y[cities[0]]], 'b-')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)
    plt.legend()
    plt.grid(True)

    return


tsp = TSP(N=20, data_path='data/pr124.tsp', beta_min=1, beta_max=20000000, Ns=10)
tour_before = tsp.current_tour
H_before = tsp.compute_energy_alternative(tour_before)
plot_tour(tour_before, title=f'Initial tour H = {H_before}')


H_list = tsp.anneal()
H_after = tsp.H_current
plot_tour(tsp.current_tour, title=f'Solution H = {H_after}')

# plt.show()

fig = plt.figure()
iters = np.arange(0, len(H_list), 1)
plt.plot(iters, H_list)
plt.show()
