import numpy as np
import random

"""
Tools for solving the tsp problem
"""


def distance_matrix(coords):
    """Create distance matrix d_ij which contains the 2D-distances
    between point i and j. 
    """
    d_ij = np.sqrt(np.square(coords[:, np.newaxis] - coords).sum(axis=2))

    return d_ij


def generate_rand_cities(N):
    
        x = np.random.rand(N)
        y = np.random.rand(N)
        return np.column_stack((x, y))
    
    
def get_nearest_neighbour_solution(distance_matrix):
    
    d_ij = distance_matrix
    
    # choose random city to start
    city = random.randrange(len(d_ij))
    result = [city]
    
    # list of cities which should be visited
    cities_to_visit = list(range(len(d_ij)))
    cities_to_visit.remove(city)
    
    # visit all cities and find the nearest
    while cities_to_visit:
        # generate touples (distance, #city) and always look for the one with smalles distance
        nearest_city = min([(d_ij[city][j], j) for j in cities_to_visit], key=lambda x:x[0])
        city = nearest_city[1]
        cities_to_visit.remove(city)
        result.append(city)
    
    
    return result
    