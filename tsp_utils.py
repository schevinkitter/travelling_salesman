import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

"""
Toolbox for solving the tsp problem
"""


def distance_matrix(coords):
    """Create distance matrix d_ij which contains the 2D-distances
    between point i and j. 
    """
    d_ij = np.sqrt(np.square(coords[:, np.newaxis] - coords).sum(axis=2))

    return d_ij


def generate_rand_cities(N: int):
    """Create random city locations in 2D

    Args:
        N (int): number of random cities

    Returns:
        np.ndarray: x and y coordinates as columns
    """
    x = np.random.rand(N)
    y = np.random.rand(N)
    return np.column_stack((x, y))


def get_nearest_neighbour_solution(distance_matrix):
    """Orders cities initially after nearest neighbour distances
    """

    d_ij = distance_matrix
    # choose random city to start
    city = random.randrange(len(d_ij))
    nn_ordering = [city]

    # list of cities which should be visited
    cities_to_visit = list(range(len(d_ij)))
    cities_to_visit.remove(city)

    # look for the next neares city
    while cities_to_visit:
        # generate touples (distance, #city) and always look for the one with smalles distance
        nearest_city = min([(d_ij[city][j], j)
                           for j in cities_to_visit], key=lambda x: x[0])
        city = nearest_city[1]
        cities_to_visit.remove(city)
        nn_ordering.append(city)

    return nn_ordering


def get_rand_solution(distance_matrix):
    # just orders the randomly generated coordinates from 1 up to the number of cities
    return [i for i in range(len(distance_matrix))]


def load_tsp_problem(path: str):
    """Can be used to load existing tsp problems with 'known' solution as .tsp files. 

    Args:
        path (str): file path of the .tsp file

    Returns:
        initial_tour: coordinate array of cities
    """
    import tsplib95
    data = tsplib95.load(path)

    node_coords = data.node_coords

    x = np.array([c[0] for c in node_coords.values()])
    y = np.array([c[1] for c in node_coords.values()])
    initial_tour = np.column_stack((x, y))

    return initial_tour


def animation_tsp(tour_history, energy_history, city_locations):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    line1, = ax1.plot([], [], lw=2)
    line2, = ax2.plot([], [], lw=2)

    def init():
        x, y = city_locations.T
        h, iters = energy_history, [i for i in range(len(energy_history))]

        ax1.plot(x, y, 'o', color='black')

        ax2.set_xlim(0, len(energy_history))
        ax2.set_ylim(0, max(energy_history) + 0.1*max(energy_history))

        line1.set_data([], [])
        line2.set_data([], [])
        return line1, line2

    def update(frame):
        x = [city_locations[i, 0]
             for i in tour_history[frame] + [tour_history[frame][0]]]
        y = [city_locations[i, 1]
             for i in tour_history[frame] + [tour_history[frame][0]]]

        h = energy_history[:frame]
        iters = list(range(frame))
        line1.set_data(x, y)
        line2.set_data(iters, h)
        return line1, line2

    ani = FuncAnimation(fig, update, frames=range(0, len(tour_history), len(tour_history)//1000),
                        init_func=init, interval=80, repeat=False)

    plt.show()

    return


def show_endresult(shortest_tour, energy_history, city_locations):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    x, y = city_locations.T

    x_coords = [city_locations[i, 0] for i in shortest_tour + [shortest_tour[0]]]
    y_coords = [city_locations[i, 1] for i in shortest_tour + [shortest_tour[0]]]

    ax1.plot(x, y, 'o', color='black')
    ax1.set_ylabel(r'$y$')
    ax1.set_xlabel(r'$x$')
    ax1.set_title('shortest route')
    ax1.plot(x_coords, y_coords)
    
    ax2.plot([i for i in range(len(energy_history))], energy_history)
    ax2.set_xlim(0, len(energy_history))
    ax2.set_ylim(0, max(energy_history) + 0.1*max(energy_history))
    ax2.set_ylabel(r'$H$')
    ax2.set_xlabel(r'iterations')
    
    # Setting xticks
    # Getting automatic ticks
    xticks = list(ax2.get_xticks())
    xticks.pop()
    # Adding one single tick to the automatic ticks
    xticks[-1] = len(energy_history)
    # Setting xticks
    ax2.set_xticks(xticks)
    ax2.axhline(y = min(energy_history), color='green', linestyle='--', label = r'$H_{min}$')
    plt.show()
    return
