from algorithms import a_star, a_star_v2, a_star_v3
from utilities import helper
import numpy as np
def CPUThreaded(algorithm='a_star', graph=None, gridArray=None, start=None, goal=None):
    path = []
    cameFrom = {}
    costSoFar = {}
    if (algorithm == 'a_star'):
        print('----- Running A* Pathfinding (CPU Threaded) -----')
        # cameFrom, costSoFar = a_star.search(graph, start, goal)
        # cameFrom, costSoFar = a_star_v2.search(graph, start, goal)
        cameFrom, costSoFar = a_star_v3.searchV2(gridArray, start, goal)
        # print(cameFrom)
        # print(np.histogram(costSoFar))
        # path = helper.reconstructPath(cameFrom, start, goal) 
        path = helper.reconstructPathV2(cameFrom, start, goal) 
    else:
        print("No implementation of search algorithm")
    # print('Came from (Hashmap): ', cameFrom)
    return (costSoFar, path)