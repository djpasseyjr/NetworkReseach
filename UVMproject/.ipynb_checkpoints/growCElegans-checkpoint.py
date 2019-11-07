from specializeGraph import *
from matplotlib import pyplot as plt
import networkx as nx
import pickle

def getA(i):
    # Load sample graph from 'celegans_samp_ i .csv' into an
    # adjacency matrix
    
    f = open('samples/celegans_samp_ '+ str(i) +' .csv','rb')
    l = f.read().split('\n')
    l = [s.split(',') for s in l[1:-1]]
    l = [(int(s[0]),int(s[1])) for s in l]
    G = nx.DiGraph(l)
    return nx.adjacency_matrix(G).toarray()

# End

def main():
    j          = 10
    i          = 1
    num_graphs = 1000
    attempts    = 0

    # For each sample graph make 15 attempts 
    # to specialize it to the correct size:

    while i <= num_graphs:

        G        = getA(i)    # Load sample i
        numNodes = G.shape[0] # Number of nodes in graph
        baseSize = 3          # Initial base size (constant)
        bases    = []         # List of bases used to store later
        growth   = 0          # Number of times specializeGraph was called



        # While the graph is not big enough

        # Begin Attempt
        while numNodes <= 270 and numNodes > 10:

            # I added the condition that the graph must be bigger than 10 nodes 
            # because some graphs would shrink down to nothing and loop forever

            # Take a random 90% of the nodes
            base = np.random.choice(range(numNodes),size=baseSize,replace=False)

            # Attempt to grow the graph
            try:
                G = specializeGraph(G,list(base))
                bases.append(base) # Store base

            except Exception:
                pass

            numNodes = G.shape[0]       # Update the number of nodes
            baseSize = int(numNodes*.9) # Compute 90% of the number of nodes
            growth   += 1               # Increase specializeGraph call count

            if growth > 200:
                # If specialize graph is called too many times, exit loop
                numNodes = 10

        # End Attempt
        attempts += 1




        if numNodes <= 350 and numNodes > 10:
            # If the number of nodes is not too big, save the matrix

            # Save matrix as csv 
            np.savetxt("gen_celegans("+str(j)+")/gen_celegans"+str(i)+".csv", 
                       G, delimiter=",",fmt='%1e')

            # Store the base sets.
            f = open("gen_base("+str(j)+")/base"+str(i)+".pkl",'wb')
            pickle.dump(bases,f)
            f.close()


            i       += 1 # Move on to next sample
            attempts = 0 # Reset attempts


        elif attempts > 15:
            # If too many attempts were made, save an empty matrix

            # Create empty matrix and empty base
            G = np.array([[0]])
            base = []

            # Save matrix
            np.savetxt("gen_celegans("+str(j)+")/gen_celegans"+str(i)+".csv", G, delimiter=",",fmt='%1e')

            # Save base set
            f = open("gen_base("+str(j)+")/base"+str(i)+".pkl",'wb')
            pickle.dump(bases,f)
            f.close()


            i       += 1 # Move on to next sample
            attempts = 0 # Reset attempts

if __name__ == '__main__':
    main()