import numpy as np
import networkx as nx
from scipy.sparse import dok_matrix
from scipy.linalg import block_diag
import itertools
import matplotlib.pyplot as plt
######## WORK TO BE DONE ##########
"""
For large matrices we get memory errors, I've found that the upper bound on the number of nodes it around 
13,000, after that the ndarray is no longer a good tool to use to store the information about the network.
Thankfully, these graphs we are working with are fairly sparse so they should be able to be implemented in
a sparse matrix.
"""
###################################

class DirectedGraph:
    '''
    A class that creates a directed graph, mainly to be used to specialize a graph

    Attributes:
        A (Square, ndarray): the adjecency matrix of a directed graph where A(i,j) is node i receiving from node j
        n (int): the number of nodes in the graph
        labels (list(str)): list of labels assigned to the nodes of the graph
        labeler (dict(int, str)): maps indices to labels
        indexer (dict(str, int)): maps labels to indices
    '''
    def __init__(self, A, dynamics, labels=None):
        '''
        Parameters: 
            A ((n,n) ndarray): The asjecency matrix to a directed graph where
                A[i,j] is node i receiving from node j
            Labels (list(str)): labels for the nodes of the graph, defaults to 0 indexing
        '''
        n,m = A.shape
        if n != m :
            raise ValueError('Matrix not square')
        if np.diag(A).sum() != 0:
            raise ValueError('Some nodes have self edges')
        # Determine how that labels will be created
        # defaults to simple 0 indexing
        if type(labels) == type(None):
            labels = [f'{i}' for i in range(n)]
        elif type(labels) != type(list()) or len(labels) != n or type(labels[0]) != type(str()):
            raise ValueError('labels must be a string list of length n')

        self.A = A
        self.n = n
        self.f = dynamics
        self.indices = np.arange(n)
        self.labeler = dict(zip(np.arange(n), labels))
        self.indexer = dict(zip(labels, np.arange(n)))
        # we use the original indexer when we look at dynamics on the network
        # this dict doesn't change under specialization
        self.original_indexer = self.indexer.copy()
    
    def set_dynamics(self):
        """
        Using a matrix valued function, set the dynamics of the network

        Implicit Parameters:
            f (nxn matrix valued function): this discribes the independent influence the jth node has on the ith node
                it will use the format for the position i,j in the matrix, node i receives from node j
        Returns:
            F (n-dimensional vector valued function): this is a vector valued function that takes in an ndarray of
                node states and returns the states of the nodes at the next time step
        """
        # here we unpack the different parts of the dynamic function
        a,f,c = self.f
        F = np.array([None]*self.n)
        for i in range(self.n):
            o_i = self.origination(i)
            # for each node we create a component function that works much like matrix multiplication
            F[i] = lambda x: np.sum([self.A[i,j] * f[o_i,self.origination(j)](x[j]) for j in range(self.n)])
        # we return a vector valued function that can be used for simple iteration
        return lambda x: [a[self.origination(k)]*x[k] + F[k](x) + c[self.origination(k)] for k in range(self.n)]

    def origination(self, i):
        """
        Returns the original index, associated with the matrix valued dynamics function, of a given node index

        Parameters:
            i (int): the current index of a given node in self.A
        Returns:
            o_i (int): the original index of i
        """
        label = self.labeler[i]
        # find the first entry in the node's label
        temp_ind = label.find('.')
        # if it is not the original label we use temp_ind
        if temp_ind != -1:
            label = label[:temp_ind]
        return self.original_indexer[label]

    
    def iterate(self, iters, initial_condition, graph=False):
        """
        Model the dynamics on the network for iters timesteps given an intial condition

        Parameters
            iters (int): number of timsteps to be simulated
            initial_condition (ndarray): initial conditions of the nodes
            graph (bool): will graph states of nodes over time if True
        Returns:
            x (ndarray): the states of each node at every time step
        """
        # grab the iterative funciton
        F = self.set_dynamics()
        # initialize an array with the initial condition
        x = [initial_condition]
        for _ in range(iters):
            # we pass into the function F the last value of our nodes
            x.append(F(x[-1]))
        
        x = np.array(x)
        
        if graph:
            domain = np.arange(iters+1)
            for i in range(self.n):
                plt.plot(domain, x[:,i], label=self.labeler[i], lw=2)
            plt.xlabel('Time')
            plt.ylabel('Node Value')
            plt.title('Network Dynamics')
            plt.legend()
            plt.show()
        
        return x

    
    def specialize_graph(self, base, verbose=False):
        """
        Given a base set, specialize the adjacency matrix of a network

        Parameters:
            base (list, int or str): list of base nodes, the other nodes will become the specialized set
            verbose (bool): print out key information as the code executes
        """
        # if the base was given as a list of nodes then we convert them to the proper indexes
        if type(base[0]) == str:
            for i, k in enumerate(base):
                base[i] = self.indexer[k] 
        elif type(base[0]) != int:
            if not np.issubdtype(base[0], np.integer):
                raise ValueError('base set must be either a list of labels or a list of indices')
        if type(base) != list:
            base = list(base)
        if len(base) > self.n:
            raise ValueError('base list is too long')

        base_size = len(base)
        # permute the matrix so the base set comes first
        self.baseFirst(base)

        # initialize the new specialized matrix as a diagonal matrix with the base set in the first slot
        B = self.A[:base_size, :base_size].copy()
        diag = [B]

        # find the strongly connected components of the network
        smallA, comp = self.compress_graph(base_size)

        pressed_paths = self.findPathsToBase(smallA, base_size, comp)

        n_nodes = base_size
        links = []

        for path in pressed_paths:
            components = [comp[k] for k in path]
            paths = self.pathCombinations(components)
            # print(f'this is after path combinations\n{paths}\n')
            
            comp_to_add = [self.A[[self.indexer[k] for k in c], :][:, [self.indexer[k] for k in c]] for c in components[1:-1]].copy()
            for p in paths:
                diag += comp_to_add
                links += self.linkAdder(p, n_nodes, components)
                n_nodes += sum(map(len, comp_to_add))

        

        # we create this diag labeler which will associate an index of the list diag to 
        # a strongly connected component set, we assume the matrices in diag are not permuted from that 
        # which we find in self.A
        diag_labeler = {}
    
        for k in comp:
            # for each strongly connected component we find where in diag this compenents exists
            ind = np.array([self.indexer[i] for i in comp[k]])
            # if the indices are within the base set we don't consider it
            if np.all(ind < len(base)): continue
            # we pull a sub matrix from self.A that corrosponds to the strongly connected component
            temp_block = self.A[ind][:,ind]
            # we check each element of diag to see if it matches the component
            for i, compt in enumerate(diag):
                if np.all(compt == temp_block):
                    diag_labeler[i] = comp[k]

        # here we update the labeler so that it will label the newly created nodes
        step = base_size
        for i in range(1,len(diag)):
            comp_len = diag[i].shape[0]
            for k in range(comp_len):
                self.labeler[step + k] = diag_labeler[i][k] + f'.{i}'
            step += comp_len
        
        self.update_indexer()
    
        if verbose:
            print(f'This is the original matrix:\n{self.A}\n')
        
        # create the new specialized matrix
        S = block_diag(*diag)
        for l in links: S[l] = 1
        self.A = S
        self.indices = np.arange(n_nodes)
        self.n = self.A.shape[0]

        if verbose:
            print(f'Strongly connected components:\n {comp}\n')
            temp_paths = []
            for path in pressed_paths:
                temp_paths.append([comp[k] for k in path])
            print(f'Paths from base node to base node:\n {temp_paths}\n')
            print(f'Number of nodes in the specialized matrix:\n {n_nodes}\n')

        return

    def update_indexer(self):
        """
        This function assumes that self.labeler is correct in its labeling:
        it reassigns the labels to the correct indices
        """
        indices = self.labeler.keys()
        labels = self.labeler.values()
        self.indexer = dict(zip(labels, indices))
        return

    def update_labeler(self):
        """
        This function assumers that self.indexer is correct in its indexing:
        it reassigns the indices to the correct labels
        """
        labels = self.indexer.keys()
        indices = self.indexer.values()
        self.labeler = dict(zip(indices, labels))
        return

    def baseFirst(self, base):
        """
        Permutes the A matrix so that the base set corrosponds to the beginning set of 
        rows and columns in A

        Parameters:
            base (list, int): a list of the indices of the base set
        
        Returns:
            None
        """
        n = self.A.shape[0]
        # find the indices of the strongly connected set and append them the the end of the base set to for the permutation of A
        to_specialize = [i for i in self.indices if i not in base]
        permute = base + to_specialize

        # Change the labeler to reflect the permutation and update the indexer
        self.labeler = {i : self.labeler[permute[i]] for i in range(self.n)}
        self.update_indexer()

        # This will rearrange the indices to put the base set first
        pA = self.A[permute,:]
        pA = pA[:,permute]

        self.A = pA
        return

    def compress_graph(self, base_size):
        """
        Creates a new matrix smallA that is the compressed adjacency matrix of A,
        each strongly connected component is represented as a single node

        Parameters:
            base_size (int): number of nodes in the base set
        
        Returns:
            smallA (ndarray, square): compressed adjacency matrix of A
            comp (dict, int: list(str)): a labling dictionary maping each node of the compressed 
                                         graph to the set of nodes it represents
        """
        # grab a portion of the matrix to find the strongly connected components
        spec = self.A[base_size:, base_size:]

        # we use nx to create a directed graph of this part
        spec_graph = nx.DiGraph(spec.T)

        # use nx to find the strongly connected components of specG
        SCComp = [np.array(list(c)) for c in nx.strongly_connected_components(spec_graph)]
        num_comp = len(SCComp)
        N = base_size + num_comp

        # comp will be a dictionary mapping base nodes to components
        # comp will keep track of the labels of the components
        comp = {}
        for i in range(base_size):
            comp[i] = [self.labeler[i]]
        
        for i in range(num_comp):
            comp[i+base_size] = [self.labeler[k + base_size] for k in SCComp[i]]

        # smallA will for our compressed A matrix lumping together all the strongly connected components
        smallA = np.zeros((N,N))
        smallA[:base_size, :base_size] = self.A[:base_size, :base_size]

        for i in range(base_size, N):
            i_indices = [self.indexer[k] for k in comp[i]]

            smallA[:base_size, i] = (self.A[:base_size, i_indices].sum(axis=1) != 0)*1.
            smallA[i, :base_size] = (self.A[i_indices, :base_size].sum(axis=0) != 0)*1.

            for j in range(base_size, i):
                j_indices = [self.indexer[k] for k in comp[j]]

                smallA[i,j] = (not (self.A[i_indices, :][:, j_indices] == 0).all())*1.
                smallA[j,i] = (not (self.A[j_indices, :][:, i_indices] == 0).all())*1.
        return smallA, comp
    
    def findPathsToBase(self, smallA, base_size, comp):
        """
        Finds all the paths between the base nodes that pass through the specialization set in the 
        compressed graph

        Parameters:
            smallA (ndarray): a compressed adjecency matrix
            base_size (int): number of nodes in the base set
        
        Returns:
            pressed_paths (list, list(str)): list of paths that pass through the specialization set 
        """
        _, N = smallA.shape
        pressed_paths = []

        # find the path between every pair of nodes
        for b1 in range(base_size):
            for b2 in range(base_size):
                # remove all other base ndes from the graph so we only find
                # paths through the specialization set
                if b1 == b2:
                    # in this case we look for cycles
                    mask = [b1] + list(range(base_size, N))
                    new_size = len(mask) + 1
                    reducedA = np.zeros((new_size, new_size))
                    reducedA[:-1, :-1] = smallA[mask,:][:, mask]
                    # remove indoing edges from the base node and add to new node
                    reducedA[-1,:] = reducedA[0,:]
                    reducedA[0,:] = np.zeros(new_size)
                    G = nx.DiGraph(reducedA.T)
                    # find paths from the base node to the new node 
                    # which is the same as finding cycles in this case
                    paths = list(nx.all_simple_paths(G,0,new_size-1))
                
                else:
                    mask = [b1,b2] + list(range(base_size, N))
                    reducedA = smallA[mask,:][:,mask]
                    # remove base node interations
                    reducedA[:2,:2] = np.zeros((2,2))
                    G = nx.DiGraph(reducedA.T)
                    paths = list(nx.all_simple_paths(G,0,1))

                # process the paths
                for p in paths:
                    if p != []:
                        if b1 == b2:
                            p = np.array(p) + base_size - 1
                        else:
                            p = np.array(p) + base_size - 2
                        p[[0,-1]] = [b1,b2]
                        pressed_paths.append(p)
        return pressed_paths

    def pathCombinations(self, components):
        """
        Given a path through the connected components of A, find every 
        unique combination of edges between the components that can be followed
        to complete the given path

        Parameters:
            components (list, ):
        """
        link_opt = []
        path_length = len(components)
        # n_nodes will keep track of the number of nodes in each branch
        n_nodes = 1

        for i in range(path_length - 1):
            rows, cols = np.where(self.A[[self.indexer[k] for k in components[i+1]],:][:, [self.indexer[k] for k in components[i]]] == 1)

            if i == 0:
                cols += [self.indexer[k] for k in components[0]]
                rows += n_nodes
            elif i == path_length - 2:
                rows += [self.indexer[k] for k in components[-1]]
                cols += n_nodes - len(components[i])
            else:
                rows += n_nodes
                cols += n_nodes - len(components[i])
            edges = zip(rows,cols)
            n_nodes += len(components[i+1])
            link_opt.append(edges)
        
        all_paths = [list(P) for P in itertools.product(*link_opt)]
        return all_paths
    
    def linkAdder(self, path, n_nodes, components):
        """
        Produces the link needed to add a branch of stringly connected components to a graph with n_nodes

        Parameters:
            path (list, tuple): edges between component nodes
            n_nodes (int): number of nodes in the original graph
            components (list, list()): 
        """
        links = []
        path_length = len(path)

        for i in range(path_length):
            if i == 0:
                links.append((path[i][0] + n_nodes - 1, path[i][1]))
            elif i == path_length - 1:
                links.append((path[i][0], path[i][1] + n_nodes - 1))
            else:
                links.append((path[i][0] + n_nodes - 1, path[i][1] + n_nodes - 1))
        
        return links