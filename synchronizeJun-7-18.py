import numpy as np
from matplotlib import pyplot as plt
import time

def F(x):
    #The logistic map with lambda = 1
    return x**2 - x

def iterate(f,n,x0):
    #Returns each f^i(x0) for
    # i = 1,2,...,n
    
    orbit = [x0]
    x = x0
    for i in range(n):
        x = f(x)
        orbit.append(x)
    return orbit

def smallWorld(nNodes,p=0.1):
    """ Generates adjacency matrix for a small world
        class network
        
        Params:
        numNodes : number of nodes in network
        p : 
    
    """
    #Create a ring
    A = np.diag(np.ones(nNodes-1),1)
    A[0,-1] = 1
    A = A+A.T
    
    for i in range(nNodes):
        for j in range(nNodes):
            if i!=j:
                if np.random.rand() > (1-p):
                    A[i,j] = 1
                    A[j,i] = 1
    return A
            
def synchronize(A,nodes_0,d=.1,iters=40):
    
    """ Parameters
        ---------
        A (nxn ndarray) graph adjacency matrix
        nodes_0 (1xn ndarray) node initial states
        d (float) dampening of network effects
        iters (int) number of iterations

        Returns
        -------
        orbit (iters x nNodes ndarray)
    """
    nNodes = A.shape[0]
    #Randomize node initial values and compute Laplacian
    nodes = nodes_0
    deg = A.sum(axis=0)
    L = np.diag(deg) - A

    #Define discrete time network dynamics
    def netwDyn(x):
        return F(x) + d*np.dot(L,x)/np.mean(deg)

    #Find the orbits of each node
    orb = np.array(iterate(netwDyn,iters,nodes))

    return orb

def plotOrbits(orb):
    nNodes = orb.shape[1]

    if nNodes >= 5:
    #Plot 5 random orbits
        r0,r1,r2,r3,r4 = np.random.choice(range(nNodes),size=5,replace=False)
        plt.plot(orb[:,r0])
        plt.plot(orb[:,r1])
        plt.plot(orb[:,r2])
        plt.plot(orb[:,r3])
        plt.plot(orb[:,r4])
        plt.show()
    
    else:
        #Plot the first two orbits
        plt.plot(orb[:,0])
        plt.plot(orb[:,1])
        plt.show()
