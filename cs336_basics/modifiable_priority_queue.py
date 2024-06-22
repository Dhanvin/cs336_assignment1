from typing import List, Tuple, Any, Dict
import unittest


# Positional helper functions for heap
def _parent_idx(pos: int):
    return (pos - 1) // 2


def _left_child_idx(pos: int):
    return 2 * pos + 1


def _right_child_idx(pos: int):
    return 2 * pos + 2


class ModifiablePriorityQueue:  # Min-Heap: Lower priority is better
    """
    Assumes that newly added nodes have low count (low priority)
    Allows adding Tuples, where all but the last element determine the priority and
    the last element references the item
    """

    def __init__(self, tuple_size: int):
        self._heap: List[Tuple[int, bytes, Any]] = (
            []
        )  # Initialize an empty list to serve as the heap
        self._item_finder: Dict[Any:int] = (
            {}
        )  # Dictionary to keep track of task positions in the heap

        # The idx into the tuple in _heap corresponding to the item in item-finder
        self.item_idx = tuple_size - 1

    def contains(self, task):
        return task in self._item_finder

    def _sift_up(self, pos):
        """
        Move the item at `pos` up the heap until it's at its correct position.

        Parameters:
        - pos: The position of the item to sift down.
        """
        newitem = self._heap[pos]
        while pos > 0:
            parentpos = _parent_idx(pos)  # Calculate parent position in the heap
            parent = self._heap[parentpos]  # Get parent item
            if parent >= newitem:  # If parent is less than or equal to new item, break
                break
            # Move parent down the heap
            self._heap[pos] = parent
            self._item_finder[parent[self.item_idx]] = (
                pos  # Update item finder for parent task
            )
            pos = parentpos  # Move pos to parent position
        self._heap[pos] = newitem  # Place new item at its correct position in the heap
        self._item_finder[newitem[self.item_idx]] = (
            pos  # Update item finder for new item task
        )

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
            if rightpos < endpos and not self._heap[childpos] > self._heap[rightpos]:
                childpos = rightpos

            # If newitem is bigger than the larger child, exit the loop
            if newitem >= self._heap[childpos]:
                break

            # Move the smaller child up.
            self._heap[pos] = self._heap[childpos]
            self._item_finder[self._heap[childpos][self.item_idx]] = pos
            pos = childpos
            childpos = _left_child_idx(pos)

        # The leaf at pos is empty now. Put newitem there and bubble it up
        # to its final resting place (by sifting its parents down).
        self._heap[pos] = newitem
        self._item_finder[newitem[self.item_idx]] = pos

    def add_task(self, task, priority=0):
        """
        Add a task with its priority to the priority queue.

        Parameters:
        - task: The task to add.
        - priority: The priority associated with the task.
        """
        if task in self._item_finder:
            return self.change_priority(
                task, priority
            )  # If task exists, change its priority
        entry = (priority, task)  # Create entry with priority and task
        pos = len(self._heap)  # Current position is at the end of the heap
        self._heap.append(entry)  # Add entry to the heap
        self._item_finder[task] = pos  # Record task's position in item_finder
        self._sift_up(pos)  # Sift the new task up to its correct position

    def pop_task(self):
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
            returnitem = self._heap[
                0
            ]  # Get the root of the heap (item with lowest priority)
            self._heap[0] = lastelt  # Move last element to the root
            self._item_finder[lastelt[self.item_idx]] = (
                0  # Update item finder for last element task
            )
            self._sift_down(0)  # Sift the root down to its correct position
        else:
            returnitem = lastelt  # If heap becomes empty, return the last element
        del self._item_finder[returnitem[self.item_idx]]  # Delete task from item finder
        return returnitem  # Return the task

    def change_priority(self, task, new_priority: Tuple):
        """
        Change the priority of an existing task in the priority queue.

        Parameters:
        - task: The task whose priority needs to be changed.
        - new_priority: The new priority for the task.
        """
        assert self.item_idx == len(new_priority)
        pos = self._item_finder[task]  # Get the position of the task in the heap
        self._heap[pos] = new_priority + (task,)  # Update the priority of the task
        self._sift_up(pos)  # Sift the task down to maintain heap order
        self._sift_down(pos)  # Sift the task up to maintain heap order

    def __len__(self):
        return len(self._heap)

    @classmethod
    def heapify(cls, items):
        """
        Build a heap from a list of items using Floyd's algorithm.
        """
        assert len(items) > 0, "Cannot heapify with empty list of items"
        tuple_size = len(items[0])
        heap = cls(tuple_size)
        heap._heap = items
        heap._item_finder = {
            prioritized_task_tuple[1]: idx
            for idx, prioritized_task_tuple in enumerate(items)
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
        pq = ModifiablePriorityQueue(2)
        pq.add_task("task1", 5)
        self.assertEqual(len(pq), 1)
        self.assertEqual(pq._heap[0], (5, "task1"))

    def test_initialize_unordered(self):
        items = [(3, "task3"), (1, "task1"), (4, "task4"), (2, "task2"), (5, "task5")]
        pq = ModifiablePriorityQueue.heapify(items)
        self.assertEqual(len(pq), 5)
        self.assertEqual(pq.pop_task(), (5, "task5"))
        self.assertEqual(pq.pop_task(), (4, "task4"))
        self.assertEqual(pq.pop_task(), (3, "task3"))
        self.assertEqual(pq.pop_task(), (2, "task2"))
        self.assertEqual(pq.pop_task(), (1, "task1"))

    def test_priority_modification(self):
        items = [(3, "task3"), (1, "task1"), (4, "task4"), (2, "task2"), (5, "task5")]
        pq = ModifiablePriorityQueue.heapify(items)
        pq.change_priority("task5", (-1))
        self.assertEqual(len(pq), 5)
        self.assertEqual(pq.pop_task(), (4, "task4"))
        self.assertEqual(pq.pop_task(), (3, "task3"))
        self.assertEqual(pq.pop_task(), (2, "task2"))
        self.assertEqual(pq.pop_task(), (1, "task1"))
        self.assertEqual(pq.pop_task(), (-1, "task5"))


if __name__ == "__main__":
    unittest.main()
