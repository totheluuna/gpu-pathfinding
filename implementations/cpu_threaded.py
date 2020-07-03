from search_algorithms import a_star
from search_algorithms import breadth_first 
from search_algorithms import djikstra
from helper import reconstructPath

def pathfinder(algorithm='a_star', graph=None, start=None, goal=None):
    path = []
    cameFrom = {}
    costSoFar = {}
    if (algorithm == 'a_star'):
        cameFrom, costSoFar = a_star.search(graph, start, goal)
        path = reconstructPath(cameFrom, start, goal) 
    elif (algorithm == 'breadth_first'):
        cameFrom = breadth_first.search(graph, start, goal)
    elif (algorithm == 'djikstra'):
        cameFrom, costSoFar = djikstra.search(graph, start, goal)
        path = reconstructPath(cameFrom, start, goal) 
    else:
        print("No implementation of search algorithm")
    # print('Came from (Hashmap): ', cameFrom)
    return (costSoFar, path)