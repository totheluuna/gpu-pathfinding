from models import priority_queue as pq

def heuristic(a, b):
    (x1, y1) = a
    (x2, y2) = b
    return abs(x1-x2) + abs(y1-y2)
def search(graph, start, goal):
    frontier = pq.PriorityQueue()
    frontier.put(start, 0)
    cameFrom = {}
    costSoFar = {}
    cameFrom[start] = None
    costSoFar[start] = 0

    while not frontier.empty():
        current = frontier.get()
        if current == goal:
            print('Found goal %s' % str(current))
            break
        for next in graph.neighbors(current):
            newCost = costSoFar[current] + graph.cost(current, next)
            if next not in costSoFar or newCost < costSoFar[next]:
                costSoFar[next] = newCost
                priority = newCost + heuristic(goal, next)
                frontier.put(next, priority)
                cameFrom[next] = current

    return cameFrom, costSoFar