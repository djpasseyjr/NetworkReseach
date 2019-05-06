import specializer as s
import numpy as np
from importlib import reload

if __name__ == "__main__":
    reload(s)
    A = np.array([[0,0,0,0,0,0,1],[1,0,1,1,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0]
    ,[0,0,1,0,0,0,0],[0,0,0,1,1,0,1],[0,0,0,1,0,1,0]])
    labels = ["a", "b", "c", 'd','e','f','g']
    # G = s.DirectedGraph(A, labels)

    # G.specialize_graph(["a",'e'], verbose=True)
    B = np.array([[0,1,1],[0,0,1],[1,0,0]])
    sig = lambda x: 1 / (1 + np.exp(-1*x))
    f = np.array([[lambda x: 0*x, sig, lambda x: -2*sig(x)],
                    [lambda x: 0*x,lambda x: 0*x, sig],
                    [sig, lambda x: 0*x, lambda x: 0*x]])
    a = np.array([5/10, 3/10, 9/10])
    c = np.array([7/2, 5/4, 1/4])
    labels = ['x1', 'x2', 'x3']
    G = s.DirectedGraph(B, (a,f,c), labels=labels)
    G.iterate(70, [1,2.5,3], graph=True)
    G.specialize_graph(['x2', 'x3'])
    G.iterate(70, np.random.random(G.n), graph=True)

