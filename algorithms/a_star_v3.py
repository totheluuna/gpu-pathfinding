from models import priority_queue_v2 as pq
import numpy as np

def passable(grid, tile):
    x,y = tile
    return grid[tile] == 1
def inBounds(grid, tile):
    (x, y) = tile
    return 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]
def getNeighbors(grid, tile):
    (x, y) = tile
    results = []
    possibleNeighbors = [(x+1,y), (x,y-1), (x-1,y), (x,y+1)]
    for tile in possibleNeighbors:
        if passable(grid, tile) and inBounds(grid, tile):
            results.append(tile)
    if (x + y)%2 == 0: results.reverse()
    return results

def heuristic(a, b):
    (x1, y1) = a
    (x2, y2) = b
    return abs(x1-x2) + abs(y1-y2)

def search(graph, start, goal):
    openList = pq.PriorityQueue()
    closedList = pq.PriorityQueue()
    
    parentHash = np.empty((graph.width, graph.height, 2), dtype=np.int32)
    parentHash[:] = np.array([-1,-1])
    GValue = np.zeros((graph.width, graph.height), dtype=np.int32)
    HValue = np.zeros((graph.width, graph.height), dtype=np.int32)
    FValue = np.zeros((graph.width, graph.height), dtype=np.int32)
    
    openList.add(start)
    startX, startY = start
    GValue[startX, startY] = 0
    HValue[startX, startY] = heuristic(start, goal)
    FValue[startX, startY] = GValue[startX, startY] + HValue[startX, startY]

    while not openList.empty():
        current = openList.pop()
        currentX, currentY = current
        if current == goal:
            print('Found goal %s' %(str(current)))
            break
        for next in graph.neighbors(current):
            # print(next)
            nextX, nextY = next
            newG = GValue[currentX, currentY] + graph.cost(current, next)
            if (openList.inQueue(next)):
                if (newG < GValue[nextX, nextY]):
                    openList.remove(next)
            if (closedList.inQueue(next)):
                if(newG < GValue[nextX, nextY]):
                    closedList.remove(next)
            if (not openList.inQueue(next)) and (not closedList.inQueue(next)):
                parentHash[nextX, nextY] = np.array([currentX, currentY])
                GValue[nextX, nextY] = newG
                HValue[nextX, nextY] = heuristic(next, goal)
                FValue[nextX, nextY] = GValue[nextX, nextY] + HValue[nextX, nextY]
                openList.add(next, FValue[nextX, nextY])
        closedList.add(current, FValue[current])

    return parentHash, FValue

def searchV2(grid, start, goal):
    width, height = grid.shape

    openList = pq.PriorityQueue()
    closedList = pq.PriorityQueue()
    
    parentHash = np.empty((width, height, 2), dtype=np.int32)
    parentHash[:] = np.array([-1,-1])
    GValue = np.zeros((width, height), dtype=np.int32)
    HValue = np.zeros((width, height), dtype=np.int32)
    FValue = np.zeros((width, height), dtype=np.int32)
    
    openList.add(start)
    startX, startY = start
    GValue[startX, startY] = 0
    HValue[startX, startY] = heuristic(start, goal)
    FValue[startX, startY] = GValue[startX, startY] + HValue[startX, startY]

    while not openList.empty():
        current = openList.pop()
        currentX, currentY = current
        if current == goal:
            print('Found goal %s' %(str(current)))
            break
        for next in getNeighbors(grid, current):
            # print(next)
            nextX, nextY = next
            newG = GValue[currentX, currentY] + 1 # constant 1 since grid
            if (openList.inQueue(next)):
                if (newG < GValue[nextX, nextY]):
                    openList.remove(next)
            if (closedList.inQueue(next)):
                if(newG < GValue[nextX, nextY]):
                    closedList.remove(next)
            if (not openList.inQueue(next)) and (not closedList.inQueue(next)):
                parentHash[nextX, nextY] = np.array([currentX, currentY])
                GValue[nextX, nextY] = newG
                HValue[nextX, nextY] = heuristic(next, goal)
                FValue[nextX, nextY] = GValue[nextX, nextY] + HValue[nextX, nextY]
                openList.add(next, FValue[nextX, nextY])
        closedList.add(current, FValue[current])

    return parentHash, FValue


    
