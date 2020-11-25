from algorithms import a_star_v3
from utilities import helper
import numpy as np

def GPUThreaded(algorithm='a_star', graph=None, gridArray=None, start=None, goal=None):
    path = []
    parents = {}
    FCost = {}
    if algorithm == 'a_star':
        parents, FCost = a_star_v3.search(graph, start, goal)
        path = helper.reconstructPathV2(parents, start, goal)
    else:
        print("No implementation of the search algorithm")
    return FCost, path