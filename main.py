import os, math, argparse
from implementations import cpu_threaded as cpu
from implementations import gpu_threaded as gpu
from implementations import gpu_pathfinder as gpu_path
from models import grid
from utilities import helper

import math
import numpy as np
from numba import cuda

from random import randint, seed
# seed(420696969)
seed(42069)

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
    # cost, path = cpu.CPUThreaded(
    #                 algorithm=algorithm,
    #                 graph=grid,
    #                 gridArray=gridArray,
    #                 start=start,
    #                 goal=goal
    #             )
    # GPU Threaded
    cost, path = gpu.GPUThreaded(
                    algorithm=algorithm,
                    graph=grid,
                    gridArray=gridArray,
                    start=start,
                    goal=goal
                )
    # GPU Pathfinder
    hArray = np.zeros(grid.shape, dtype=np.int32)
    TPB = 16
    threadsperblock = (TPB, TPB)
    blockspergrid_x = math.ceil(gridArray.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(gridArray.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    gpu_path.GPUPathfinder[blockspergrid, threadsperblock](
        algorithm=algorithm,
        grid=gridArray,
        start=start,
        goal=goal,
        hArray=hArray
    )
    print(hArray)

    # Reconstruct and draw the grid and the found path
    # Saves reconstruction on a text file
    helper.drawGrid3(grid, start, goal, path=path)
    # print('cost: ', cost, 'path: ', path)
if __name__ == "__main__":
    main()