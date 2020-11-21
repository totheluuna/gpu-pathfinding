from models import priority_queue_v2 as pq
import numpy as np

def heuristic(a, b):
    (x1, y1) = a
    (x2, y2) = b
    return abs(x1-x2) + abs(y1-y2)

def search(graph, start, goal):
    openList = pq.PriorityQueue()
    closedList = pq.PriorityQueue()
    parentHash = {}
    GValue = {}
    HValue = {}
    FValue = {}

    openList.add(start)
    parentHash[start] = None
    GValue[start] = 0
    HValue[start] = heuristic(start, goal)
    FValue[start] = GValue[start] + HValue[start]

    while not openList.empty():
        current = openList.pop()
        if current == goal:
            print('Found goal %s' %(str(current)))
            break
        for next in graph.neighbors(current):
            # print(next)
            newG = GValue[current] + graph.cost(current, next)
            if (openList.inQueue(next)):
                if (newG < GValue[next]):
                    openList.remove(next)
            if (closedList.inQueue(next)):
                if(newG < GValue[next]):
                    closedList.remove(next)
            if (not openList.inQueue(next)) and (not closedList.inQueue(next)):
                parentHash[next] = current
                GValue[next] = newG
                HValue[next] = heuristic(next, goal)
                FValue[next] = GValue[next] + HValue[next]
                openList.add(next, FValue[next])
        closedList.add(current, FValue[current])

    return parentHash, FValue


    
