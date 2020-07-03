import os, math, argparse
from implementations import cpu_threaded as cpu
from models import grid
from utilities import helper

def main():
    parser = argparse.ArgumentParser(description='Simulate sending locational data to the NIMPA server.')
    parser.add_argument('algorithm', type=str, help='Name of the pathfinding algorithm to use')
    parser.add_argument('--start', nargs='+', type=int, dest='start', help='Define the start point on the 2D grid.')
    parser.add_argument('--goal', nargs='+', type=int, dest='goal', help='Define the end point on the 2D grid.')
    args = parser.parse_args()
    algorithm = args.algorithm
    start = tuple(args.start)
    goal = tuple(args.goal)
    ''' 
    Calls the pathfinder function
    Input: algorithm to use, start node, end node
    Output: Cost of the Path, List of nodes to pass through
    '''
    # initialize grid
    sampleGrid = helper.createGrid(20,20,algorithm)
    # helper.drawGrid(sampleGrid, start, goal)
    
    # find the shortest path
    # parameters: algorithm, graph, start, end
    # returns the cost of the path
    # and a list of nodes that constitute the shortest path
    cost, path = cpu.CPUThreaded(
                    algorithm=algorithm,
                    graph=sampleGrid,
                    start=start,
                    goal=goal
                )
    # helper.drawGrid(sampleGrid, cost=cost)
    helper.drawGrid(sampleGrid, start, goal, path=path)
    # print('cost: ', cost, 'path: ', path)
if __name__ == "__main__":
    main()