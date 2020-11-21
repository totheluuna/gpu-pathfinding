from algorithms import a_star, a_star_v2
from utilities import helper
def CPUThreaded(algorithm='a_star', graph=None, start=None, goal=None):
    path = []
    cameFrom = {}
    costSoFar = {}
    if (algorithm == 'a_star'):
        print('----- Running A* Pathfinding (CPU Threaded) -----')
        # cameFrom, costSoFar = a_star.search(graph, start, goal)
        cameFrom, costSoFar = a_star_v2.search(graph, start, goal)
        print(cameFrom)
        path = helper.reconstructPath(cameFrom, start, goal) 
    else:
        print("No implementation of search algorithm")
    # print('Came from (Hashmap): ', cameFrom)
    return (costSoFar, path)