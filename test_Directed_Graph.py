from specializer import DirectedGraph
import numpy as np

if __name__ == "__main__":
    A = np.array([[0,0,0,0,0,0,1],[1,0,1,1,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0]
    ,[0,0,1,0,0,0,0],[0,0,0,1,1,0,1],[0,0,0,1,0,1,0]])
    labels = ["a", "b", "c", 'd','e','f','g']
    G = DirectedGraph(A, labels)

    # G.specialize_graph(["a",'e'], verbose=True)
