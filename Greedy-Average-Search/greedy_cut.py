"""
Greedy Average Search Algorithm for Max-k-Cut Problem
"""

import numpy as np
from anytree import Node


class GreedyFourierSearch:
    def __init__(self, A, k):
        # Validate inputs
        if not isinstance(k, int) or not isinstance(A, np.ndarray):
            raise ValueError("k must be an integer and A must be a numpy array")
        
        n = A.shape[0]
        if k > n:
            raise ValueError("k must be less than n")
        
        if A.shape[0] != A.shape[1]:
            raise ValueError("A must be a square matrix")
        
        if not np.all(A >= 0):
            raise ValueError("A must contain only non-negative values")
        
        if not np.allclose(A, A.T):
            raise ValueError("A must be symmetric")        
        
        # Calculate degrees of vertices
        degrees = np.sum(A, axis=1)
        
        # Get the sorted indices based on degrees
        sorted_indices = np.argsort(degrees)
        
        # Reorder the matrix A based on sorted indices
        A = A[sorted_indices, :][:, sorted_indices]
        
        self.A = A  # Reordered matrix greatly improves performance
        self.k = k
        self.n = n
        self.root = Node(("root", []), bound=float('inf'))
        self.sorted_indices = sorted_indices  # Store for later use

    def get_address(self, node):
        if node.is_root:
            return []
        return [node.name[0]] + node.parent.name[1]  # Addresses are written in reverse

    def value(self, node):
        # Check if the node is an nth child of the root
        if node.depth != self.n:
            return print(str(node.name[1]) + " tried to be evaluated rather than bounded")
        
        # Use the address directly
        z = node.name[1]
        
        result = 0
        for i in range(self.n):
            for j in range(i+1, self.n):
                if z[i] != z[j]:
                    result += self.A[i][j]
        
        return result
    
    def _bound(self, node):
        """
        Computes the upper bound for a given node.
        """
        address = node.name[1]
        l = len(address)

        if l <= 1:  # To stop unnecessary computation at the start
            return [float('inf')]
        
        if l >= self.n:
            print("Address at " + str(address) + " is " + str(l) + " long")
            return [float('inf')]
        
        # Term 0 (|m_0|= 0)
        Al = self.A[:self.n-l, :self.n-l]  # Accounting for 0-indexing
        b0x = self.k * 0.5 * np.sum(self.A)

        # Convert address to a NumPy array
        address_array = np.array(address)
        # Create the Del matrix using broadcasting
        Del = (address_array[:, None] == address_array[None, :]).astype(int)
        b0y = (-1) * 0.5 * self.k * np.sum(np.multiply(Del, self.A[self.n-l:, self.n-l:]))
        
        b0z = (-1) * 0.5 * np.sum(Al)
        
        b0 = b0x + b0y + b0z
        
        return [float(b0/self.k)]
    
    def _branch(self, node):
        # Only create one child for the root node
        if node == self.root:
            child = Node((0, [0]), parent=node)
            child.bound = float('inf')
            return

        if len(node.name[1]) >= self.n:
            print(str(node.name[1]) + " tried to have children") 
            return
        
        if len(node.name[1]) == self.n-1:
            for i in range(self.k):
                child = Node((i, [i] + node.name[1]), parent=node)
                child.bound = self.value(child)
            return

        # Create k children nodes
        for i in range(self.k): 
            child = Node((i, [i] + node.name[1]), parent=node)
            child.bound = sum(self._bound(child))
            if node:
                if child.bound > node.bound:
                    child.bound = node.bound
        return
    
    def search(self, node):
        if len(node.name[1]) == self.n:
            return [float(node.bound), node.name[1]]

        self._branch(node)
        best_child = max(node.children, key=lambda child: child.bound)
        return self.search(best_child)
    
    def solve(self):
        """Main solving method that returns partition and cut value"""
        result = self.search(self.root)
        cut_value = result[0]
        partition = result[1]
        
        # Reorder partition back to original vertex ordering
        original_partition = [0] * self.n
        for i in range(self.n):
            original_partition[self.sorted_indices[i]] = partition[i]
        
        return original_partition, cut_value
