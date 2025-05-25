import numpy as np
from anytree import Node

class LazyBranchAndBound:
    def __init__(self, A, k, initial_guess):
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
        
        if not isinstance(initial_guess, (int, float)) or initial_guess < 0:
            raise ValueError("initial_guess must be a non-negative number")
        
        # Calculate degrees of vertices
        degrees = np.sum(A, axis=1)
        
        # Get the sorted indices based on degrees
        sorted_indices = np.argsort(degrees)
        
        # Reorder the matrix A based on sorted indices
        A = A[sorted_indices, :][:, sorted_indices]
        
        self.A = A #Reordered matrix greatly improves performance.
        self.k = k
        self.n = n
        self.best_guess = (initial_guess, [])
        self.root = Node(("root", []), bound=float('inf'))

    def get_address(self, node):
        if node.is_root:
            return []
        return [node.name[0]] + node.parent.name[1] #Addresses are written in reverse to go from n down to 1.

    def _bound(self, node):
        """
        Computes the upper bound for a given node.
        For now, it returns +infinity as a dummy upper bound.

        :param node: The node for which to compute the upper bound.
        :return: The computed upper bound.
        """
        
        address = node.name[1]
        l = len(address)

        if l <= 1: #To stop unnecessary computation at the start.
            return [float('inf')]
        
        if l >= self.n: #DEBUG LINE
            print("Address at " + str(address) + " is " + str(l) + " long")
            return [float('inf')]

        #Term 1 (|m_0|= 2) 
        b2 = 0
        Al = self.A[:self.n-l, :self.n-l] #Accounting for 0-indexing
        b2 += np.sum(Al)*self.k*0.5 #These factors of 0.5 are because we're iterating over all pairs of the Fourier mode components (m_s,m_t) for all s,t.
        """
        Part of this term is cancelled out by the last part of Term 3.
        Actually quite small, maybe because n is small. 
        Spectral methods may help for large n
        """
        #Term 2 (|m_0|= 1) 
        b1 = 0
        b1r = 0
        for s in range(self.n-l):
            for t in range(1,self.k):
                for r in range(l):
                    b1r += self.A[s][r+self.n-l] * np.exp((-1)*2*np.pi * 1j * t*address[r]/self.k)
                b1 += abs(b1r)
        #
        """
        This term needs to be much more efficiently implemented.
        There might be some cancellation with the different t and r.
        HORRIBLY INEFFICIENT, at its worst when l = n/2.
        """
        #Term 3 (|m_0|= 0)
        b0x = self.k*0.5*np.sum(self.A)

        # Convert address to a NumPy array
        address_array = np.array(address)
        # Create the Del matrix using broadcasting
        Del = (address_array[:, None] == address_array[None, :]).astype(int)
        b0y = (-1)*0.5*self.k*np.sum(np.multiply(Del,self.A[self.n-l:,self.n-l:,]))
        
        b0z = (-1)*0.5*np.sum(Al) #This looks like term 1 a lot, let's cancel out later.
        
        b0 = b0x+b0y+b0z
        """
        Actually pretty good - greedy search looks promising!
        Fourier sparsity => Smoothness => Low Variance => Greedy works well. 
        """

        if b0 < 0: #DEBUG LINE, this should never be less than 0.
            print("Average is negative? Term 3 is " +str(b0) + " at " + str(address))
        
        if l == 2:
            return [b0/self.k,b1/self.k,0] #Since there's no 2-modes yet.
        
        return [b0/self.k,b1/self.k,b2/self.k] #There's no factor of half anymore, mistake carried from thesis and early calculations.
    
    def _branch(self, node):
        #Only create one child for the root node, this is the beginnings of the overall branching simplification.
        if node == self.root:
            child = Node((0, [0]),parent=node)
            child.bound = float('inf')
            return

        # Create k children nodes
        if len(node.name[1]) >= self.n: #DEBUG LINE
            print(str(node.name[1]) + " tried to have children") 
            return #Is creating extra nodes beyond depth n, this is a temporary fix.
        
        for i in range(self.k): 
            child = Node((i, [i] + node.name[1]), parent=node)

    def value(self, node):
        # Check if the node is an nth child of the root
        if node.depth != self.n: #DEBUG LINE
            return print(str(node.name[1]) + " tried to be evaluated rather than bounded")
        
        # Use the address directly
        z = node.name[1]
        
        result = 0
        for i in range(self.n):
            for j in range(i+1,self.n):
                if z[i] != z[j]:
                    result += self.A[i][j]
        
        return result

    def _prune(self, node):
        # Recursively delete all children
        for child in node.children:
            self._prune(child)
        # Detach the node from its parent and delete it
        node.parent = None
        del node

    def search(self, updates=False): # Add updates parameter with default value False
        stack = [self.root]
        best_guess_node = None
        
        while stack:  # While stack is not empty
            node = stack.pop()
            if len(node.name[1]) != node.depth: #DEBUG LINE
                print("length of " + str(node.name[1]) + " is not equal to its depth " + str(node.depth))
                
            if len(node.name[1]) < self.n:
                node.bound = sum(self._bound(node))
                if node.parent:
                    if node.bound > node.parent.bound:
                        node.bound = node.parent.bound
                if node.bound >= self.best_guess[0]:
                    self._branch(node)
                    for child in node.children:
                        stack.append(child)
                else:
                    # Prune the node and its children
                    self._prune(node)
                    
            elif len(node.name[1]) == self.n:
                # Compute value for the node. Sibling strategy yet to be implemented.
                parent = node.parent
                if parent: #This stops excessive breeding.
                    value = self.value(node)
                    if value < self.best_guess[0]:
                        self._prune(node)
                            
                    else:
                        self.best_guess = [float(value), node.name[1]]
                        if updates:  # Print only if updates is True
                            print("New Best guess:" + str(self.best_guess))
                            
                        # Remove the old best_guess node from the stack and the tree
                        if best_guess_node in stack:
                            stack.remove(best_guess_node)
                        if best_guess_node:
                            self._prune(best_guess_node)
                                
                        # Update the best_guess_node
                        best_guess_node = node
                else: 
                    print(str(node.name[1]) + " somehow exists without a parent") #DEBUG LINE
        
        return self.best_guess