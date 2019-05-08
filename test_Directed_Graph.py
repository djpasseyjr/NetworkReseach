import specializer as s
import numpy as np
from importlib import reload
import networkx as nx
import matplotlib.pyplot as plt

if __name__ == "__main__":
    reload(s)
    A = np.array([[0,0,0,0,0,0,1],[1,0,1,1,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0]
    ,[0,0,1,0,0,0,0],[0,0,0,1,1,0,1],[0,0,0,1,0,1,0]])
    labels = ["a", "b", "c", 'd','e','f','g']
    # G = s.DirectedGraph(A, labels)

    # G.specialize_graph(["a",'e'], verbose=True)
    B = np.array([[0,0,1],[1,0,0],[1,1,0]])
    sig = lambda x: np.tanh(x)
    f = np.array([[lambda x: 0*x, lambda x: 0*x, lambda x: -(1/2)*sig(x)],
                    [lambda x: 0.5*sig(x), lambda x: 0*x, lambda x: 0*x],
                    [lambda x: 0.5*sig(x), lambda x: 0.5*sig(x), lambda x: 0*x]])
    a = np.array([3/10, 3/10, 3/10])
    c = np.array([1/4, 0.025, 0])
    labels = ['x1', 'x2', 'x3']

    G = s.DirectedGraph(B, (a,f,c), labels=labels)
    G.iterate(80, np.random.random(G.n), graph=True)

    G.specialize_graph(['x2', 'x3'], verbose=False)
    G.iterate(80, np.random.random(G.n), graph=True)

    G.specialize_graph(['x1.1', 'x1.2', 'x2'], verbose=False)
    G.iterate(80, np.random.random(G.n), graph=True)

    # N = nx.DiGraph(G.A.T)
    # nx.draw_networkx_labels(N, pos=nx.spring_layout(N), labels=G.labeler)
    # nx.draw(N, pos=nx.spring_layout(N))
    # plt.draw()
    # plt.show()


