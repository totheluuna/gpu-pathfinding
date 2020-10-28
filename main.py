import os, math, argparse
from implementations import cpu_threaded as cpu
from models import grid
from utilities import helper

import math
import numpy as np

from random import randint, seed
seed(69)

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
    image = "dataset/da2-png/ht_mansion2b.png"
    # sampleGrid = helper.createGrid(20,20,algorithm)
    sampleGrid = helper.imageToGrid(image)
    helper.drawGrid2(sampleGrid, start, goal)

    dist = 0
    hypotenuse = math.sqrt(math.pow(sampleGrid.width, 2) + math.pow(sampleGrid.height,2))
    # generate random start and goal
    # ensure that the distance between the start and goal 
    # is 50% the hypotenuse of the dims of the grid 
    while dist <= 0.50*hypotenuse:
        random_x = randint(0, sampleGrid.width-1)
        random_y = randint(0, sampleGrid.height-1)
        start = (random_x, random_y)
        while start in sampleGrid.walls:
            random_x = randint(0, sampleGrid.width-1)
            random_y = randint(0, sampleGrid.height-1)
            start = (random_x, random_y)

        random_x = randint(0, sampleGrid.width-1)
        random_y = randint(0, sampleGrid.height-1)
        goal = (random_x, random_y)
        while (goal in sampleGrid.walls):
            random_x = randint(0, sampleGrid.width-1)
            random_y = randint(0, sampleGrid.height-1)
            goal = (random_x, random_y)
    
        print('start: ', start, ' goal: ', goal)
        a = np.array(start)
        b = np.array(goal)
        dist = np.linalg.norm(a-b)
    

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