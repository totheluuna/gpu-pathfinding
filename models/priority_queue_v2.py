import heapq
import itertools

class PriorityQueue:
    def __init__(self):
        self.elements = []
        self.entry_finder = {}
        self.REMOVED = '<removed-item>'
        self.counter = itertools.count()
    def empty(self):
        return len(self.elements) == 0
    def inQueue(self, item):
        return item in self.entry_finder
    def add(self, item, priority=0):
        if item in self.entry_finder:
            self.remove(item)
        count = next(self.counter)
        entry = [priority, count, item]
        self.entry_finder[item] = entry
        heapq.heappush(self.elements, entry)
    def remove(self, item):
        entry = self.entry_finder.pop(item)
        entry[-1] = self.REMOVED
    def pop(self):
        priority, count, item = heapq.heappop(self.elements)
        if item is not self.REMOVED:
            del self.entry_finder[item]
            return item
        raise KeyError('pop from an empty priority queue')
        # return heapq.heappop(self.elements)[1]
