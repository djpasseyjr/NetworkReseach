import numpy as np
from scipy import linalg as la
from matplotlib import pyplot as plt
import networkx as nx

def F(x):
    #Internal dynamics of each node
    return x

def G(x):
    #External dynamics of each node
    #(Nothing)
    return x

def iterate(x0,f,iters=100):
    """
    Returns n length orbit of x0 under f
    Params
    ------
    x0 : scalar or ndarray
    f : function
    n : positive integer
    Returns
    -------
    orbit : ndarray- orbit of x0 under f
    """
    orbit = [x0]
    x = x0
    for i in range(iters):
        x = f(x)
        orbit.append(x)
    return np.array(orbit)

def getGraphDynam(F,G,A,d=.2,normalize=False):
    """
    Computes the orbit of x under the following equation:
    x_i[k+1] = F(x_i[k])+d*SUM_j{ A_ij*[G(x_i[k])-G(x_j[k])] }
         for j = 1,2, ... n
     Parameters
     ----------
     F (function): Internal dynamics of nodes
     G (function): How nodes affect eachother
     A (nxn ndarray): Adjacency Matrix
     d (float): dampening parameter
     
     Returns
     -------
     GraphDynam (function) : dynamics on graph as described above
    """
    degr = A.sum(axis=1)*1.
    
    if normalize:
        degr[degr!=0] = 1./degr[degr!=0]
        Dinv = np.diag(degr)
        L = np.diag((degr != 0)*1.) - np.dot(Dinv,A)
    else:
        L = np.diag(degr) - A

    def GraphDynam(x):
        return F(x) + d*np.dot(L,G(x))
    
    return GraphDynam

def plotAllOrb(orb,k=0,show=True):
    #Plots every orbit
    m,n = orb.shape
    
    if k == 0:
        iteraxis = np.arange(m)
    if k < 0:
        iteraxis = np.arange(m+k,m)
    if k > 0:
        iteraxis = np.arange(k,m)
        
    
    plt.rcParams['figure.figsize'] = (10,5)
    for i in range(n):
        plt.plot(iteraxis,orb[k:,i],label="Node "+str(i+1))
    plt.xlabel("Iteration")
    plt.ylabel("Node Values")
    plt.legend()
    
    if show:
        plt.show()
        print("Node variance in last iteration: {}".format(np.var(orb[-1])))
    
def plotRandomOrbits(orb,k=0,show=True):
    #Plots ten random orbits
    m,n = orb.shape
    
    if k == 0:
        iteraxis = np.arange(m)
    if k < 0:
        iteraxis = np.arange(m+k,m)
    if k > 0:
        iteraxis = np.arange(k,m)
        
    nodes = np.random.choice(range(n),size=10,replace=False)
    plt.rcParams['figure.figsize'] = (10,5)
    for n in nodes:
        plt.plot(iteraxis,orb[k:,n],label="Node "+str(n+1))
    
    plt.xlabel("Iteration")
    plt.ylabel("Node Values")
    plt.legend()
    
    if show:
        plt.show()
        print("Variance in last orbit: {}".format(np.var(orb[-1])))
    
def netwDyn(A,d=.2,k=0,x0=None,iters=100,graph=True,F=F,G=G,normalize=False):
    """
    Plot node dynamics using the functions above
    """
    m,n = A.shape
    if graph:
        labels = {}
        for i in range(n):
            labels[i]=str(i+1)
        gr = nx.from_numpy_matrix(A.T,create_using=nx.DiGraph())
        nx.draw(gr,arrows=True,node_color='#15b01a',labels=labels)
        plt.show()
    
    if x0 is None:
        x0 = np.random.rand(n)*2-1

    GraphDyn = getGraphDynam(F,G,A,d=d,normalize=normalize)
    orbit = iterate(x0,GraphDyn,iters=iters)
    
    if n > 13:
        plotRandomOrbits(orbit,k=k)
        print("\nLast four iterations: (Nodes 1-5)")
        print(orbit[-4:,:5])
    else:
        plotAllOrb(orbit,k=k)
        print("\nLast four iterations:")
        print(orbit[-4:,:])
    
    return x0

def LAndRandLDyn(A,d=-.2,dN=-1,k=0,x0=None,x1=None,iters=100,graph=True,F=F,G=G):
    m,n = A.shape
    if graph:
        labels = {}
        for i in range(n):
            labels[i]=str(i+1)
        gr = nx.from_numpy_matrix(A.T,create_using=nx.DiGraph())
        nx.draw(gr,arrows=True,node_color='#15b01a',labels=labels)
        plt.show()
    
    plt.rcParams['figure.figsize'] = (10,5)
    plt.subplot(121)
    if x0 is None:
        x0 = np.random.rand(n)*2-1
    
    GraphDyn = getGraphDynam(F,G,A,d=d,normalize=False)
    orbit = iterate(x0,GraphDyn,iters=iters)
    
    if n > 13:
        plotRandomOrbits(orbit,k=k,show=False)
    else:
        plotAllOrb(orbit,k=k,show=False)
    plt.title("Normal")
    
    plt.subplot(122)
    if x1 is None:
        x1 = np.random.rand(n)*2-1
    
    GraphDyn = getGraphDynam(F,G,A,d=dN,normalize=True)
    NormOrbit = iterate(x0,GraphDyn,iters=iters)
    
    if n > 13:
        plotRandomOrbits(NormOrbit,k=k,show=False)
    else:
        plotAllOrb(NormOrbit,k=k,show=False)
    plt.title("Random Walk")
    plt.show()
        
    if n > 13:
        print("\nLast four iterations: (Nodes 1-5)")
        print(orbit[-4:,:5])
        print("\nRandom Walk Last four iterations: (Nodes 1-5)")
        print(NormOrbit[-4:,:5])
    else:
        print("\nLast four iterations:")
        print(orbit[-4:,:])    
        print("\nRandom Walk Last four iterations:")
        print(NormOrbit[-4:,:])
