import os
import sys

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

from grid import Grid

class WeightedGrid(Grid):
    def __init__(self, width, height):
        super().__init__(width, height)
        self.weights = {}
    def cost(self, fromNode, toNode):
        return self.weights.get(toNode, 1)