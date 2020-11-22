import os, math, argparse
from implementations import cpu_threaded as cpu
from models import grid
from utilities import helper

import math
import numpy as np

from random import randint, seed
seed(420696969)

def main():
    ''' 
    Calls the pathfinder function
    Input: algorithm to use, start node, end node
    Output: Cost of the Path, List of nodes to pass through
    '''

    parser = argparse.ArgumentParser(description='GPU Pathfinding')
    parser.add_argument('algorithm', type=str, help='Name of the pathfinding algorithm to use', default='a_star')
    args = parser.parse_args()
    algorithm = args.algorithm
    
    # initialize grid from image, selects random image from dataset
    grid, gridArray = helper.createGridFromDatasetImage('dataset/da2-png')
    print(gridArray)

    # generate random start and goal
    # ensure that the distance between the start and goal 
    # is 50% the hypotenuse of the dims of the grid 
    start, goal = helper.randomStartGoal(grid)

    
    # Find the shortest path
    # parameters: algorithm, graph, start, end
    # returns the cost of the path
    # and a list of nodes that constitute the shortest path
    cost, path = cpu.CPUThreaded(
                    algorithm=algorithm,
                    graph=grid,
                    gridArray=gridArray,
                    start=start,
                    goal=goal
                )
    # Reconstruct and draw the grid and the found path
    # Saves reconstruction on a text file
    helper.drawGrid3(grid, start, goal, path=path)
    # print('cost: ', cost, 'path: ', path)
if __name__ == "__main__":
    main()