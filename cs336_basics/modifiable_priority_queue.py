from typing import List, Tuple, Any, Dict
import functools
import unittest


# Positional helper functions for heap
def _parent_idx(pos: int):
    return (pos - 1) // 2


def _left_child_idx(pos: int):
    return 2 * pos + 1


def _right_child_idx(pos: int):
    return 2 * pos + 2

@functools.total_ordering
class HeapItem:
    def __init__(self, name: str, priority: Tuple):
        self.name = name
        self.priority = priority

    def __eq__(self, other):
        self.priority == other.priority

    def __gt__(self, other):
        self.priority > other.priority

    def __repr__(self):
        return f"item:{self.name}; priority: {self.priority}"



class ModifiablePriorityQueue:  
    """
    Max-Heap: Higher priority is better.
    Operates over HeapItem class, with defined name, priority
    Allows for efficient re-prioritization without having to call heapify repeatedly.
    """

    def __init__(self):
        # Initialize an empty list to serve as the heap
        self._heap: List[HeapItem] = []
        # Dictionary to keep track of item positions in the heap. 
        # Used for updating priorities of specific items
        self._item_finder: Dict[str:int] = {}

    def contains(self, item):
        return item in self._item_finder
    
    def get_priority(self, item):
        return self._heap[self._item_finder[item]].priority

    def _sift_up(self, pos):
        """
        Move the item at `pos` up the heap until it's at its correct position.

        Parameters:
        - pos: The position of the item to sift down.
        """
        newitem = self._heap[pos]
        while pos > 0:
            parentpos = _parent_idx(pos)  
            parent = self._heap[parentpos]  
            if parent.priority >= newitem.priority: 
                break
            # Move parent down the heap
            self._heap[pos] = parent
            self._item_finder[parent.name] = pos 

            pos = parentpos  # Move pos to parent position
        self._heap[pos] = newitem  # Place new item at its correct position in the heap
        self._item_finder[newitem.name] = pos 
        

    def _sift_down(self, pos):
        """
        Move the item at `pos` down the heap until it's at its correct position.

        Parameters:
        - pos: The position of the item to sift down.
        """
        endpos = len(self._heap)
        newitem = self._heap[pos]

        # Follow the path to a leaf, moving parents down until finding a place
        # newitem fits.
        childpos = _left_child_idx(pos)
        while childpos < endpos:
            rightpos = _right_child_idx(pos)
            # Set childpos to index of larger child.
            if rightpos < endpos and not self._heap[childpos].priority > self._heap[rightpos].priority:
                childpos = rightpos

            # If newitem is bigger than the larger child, exit the loop
            if newitem.priority >= self._heap[childpos].priority:
                break

            # Move the bigger child up.
            self._heap[pos] = self._heap[childpos]
            self._item_finder[self._heap[childpos].name] = pos
            pos = childpos
            childpos = _left_child_idx(pos)

        # The leaf at pos is empty now. Put newitem there and bubble it up
        # to its final resting place (by sifting its parents down).
        self._heap[pos] = newitem
        self._item_finder[newitem.name] = pos

    def add_task(self, name, priority=0):
        """
        Add a task with its priority to the priority queue.

        Parameters:
        - task: The task to add.
        - priority: The priority associated with the task.
        """
        if name in self._item_finder:
            return self.change_priority(
                name, priority
            )  # If task exists, change its priority
        entry = HeapItem(name, priority)  
        pos = len(self._heap)  
        self._heap.append(entry)  
        self._item_finder[name] = pos  
        self._sift_up(pos)

    def pop_task(self) -> HeapItem:
        """
        Remove and return the task with the lowest priority (root of the heap).

        Returns:
        - The task with the lowest priority.
        """
        if not self._heap:
            raise KeyError(
                "pop from an empty priority queue"
            )  # Raise error if heap is empty
        lastelt = self._heap.pop()  # Remove the last element from the heap
        if self._heap:
            # Get the root of the heap (item with lowest priority)
            returnitem = self._heap[0]  
            self._heap[0] = lastelt  # Move last element to the root
            self._item_finder[lastelt.name] = 0
            self._sift_down(0)  # Sift the root down to its correct position
        else:
            returnitem = lastelt  # If heap becomes empty, return the last element
        del self._item_finder[returnitem.name]  # Delete task from item finder
        return returnitem  # Return the task

    def change_priority(self, task, new_priority: Tuple):
        """
        Change the priority of an existing task in the priority queue.

        Parameters:
        - task: The task whose priority needs to be changed.
        - new_priority: The new priority for the task.
        """
        pos = self._item_finder[task] 
        self._heap[pos].priority = new_priority
        
        # Sift the task up and down to maintain heap order
        self._sift_up(pos)  
        self._sift_down(pos)  

    def __len__(self):
        return len(self._heap)

    @classmethod
    def heapify(cls, items: List[HeapItem]):
        """
        Build a heap from a list of items using Floyd's algorithm.
        """
        assert len(items) > 0, "Cannot heapify with empty list of items"
        heap = cls()
        heap._heap = items
        heap._item_finder = {
            heap_item.name: idx
            for idx, heap_item in enumerate(items)
        }

        # Sift_down start from the last non-leaf node backwards till start of the heap. This ensures fewest swaps during construction
        # Leaf nodes start from n//2 to end of list. So ignore them
        startpos = len(heap._heap) // 2 - 1
        for pos in range(startpos, -1, -1):
            heap._sift_down(pos)
        return heap


# -- Unit Tests ---
class TestModifiablePriorityQueue(unittest.TestCase):
    # Each test method must start with test_
    def test_add_task(self):
        pq = ModifiablePriorityQueue()
        pq.add_task("task1", (5,))
        self.assertEqual(len(pq), 1)
        # breakpoint()
        self.assertEqual(pq._heap[0].name, 'task1')
        self.assertEqual(pq._heap[0].priority, (5,))

    def test_initialize_unordered(self):
        items = [HeapItem('task3', (3,)), HeapItem('task1', (1,)), HeapItem('task4', (4,)), HeapItem('task2', (2,)), HeapItem('task5', (5,))]
        pq = ModifiablePriorityQueue.heapify(items)
        self.assertEqual(len(pq), 5)
        self.assertEqual(pq.pop_task().name, 'task5')
        self.assertEqual(pq.pop_task().name, 'task4')
        self.assertEqual(pq.pop_task().name, 'task3')
        self.assertEqual(pq.pop_task().name, 'task2')
        self.assertEqual(pq.pop_task().name, 'task1')


    def test_priority_modification(self):
        items = [HeapItem('task3', (3,)), HeapItem('task1', (1,)), HeapItem('task4', (4,)), HeapItem('task2', (2,)), HeapItem('task5', (5,))]
        pq = ModifiablePriorityQueue.heapify(items)
        pq.change_priority("task5", (-1,))
        self.assertEqual(len(pq), 5)
        self.assertEqual(pq.pop_task().name, 'task4')
        self.assertEqual(pq.pop_task().name, 'task3')
        self.assertEqual(pq.pop_task().name, 'task2')
        self.assertEqual(pq.pop_task().name, 'task1')
        final = pq.pop_task()
        self.assertEqual(final.name, 'task5')
        self.assertEqual(final.priority, (-1,))


if __name__ == "__main__":
    unittest.main()
