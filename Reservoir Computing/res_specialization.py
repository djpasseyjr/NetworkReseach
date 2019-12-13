"""
Functions used to examine reservoir specialization

"""
from ResComp import *
from specializeGraph import *
from sparse_specializer import *
import copy

def lorentz_deriv(t0, X, sigma=10., beta=8./3, rho=28.0):
    """Compute the time-derivative of a Lorenz system."""
    (x, y, z) = X
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]
# end

def lorenz_equ(x0=[-20, 10, -.5], begin=0, end=60, timesteps=60000):
    """Use solve_ivp to produce a solution to the lorenz equations""" 
    t = np.linspace(begin,end,timesteps)
    train_t = t[:len(t)//2]
    u = integrate.solve_ivp(lorentz_deriv, (begin,end),x0, dense_output=True).sol
    return t, train_t, u
# end

def toterror(u,pre):
    """Sum of ||u_i - pre_i||_2 for all i"""
    return np.sum(np.sum((u - pre)**2, axis=0)**.5)
# end

def L2error(u,pre):
    """Mean squared error"""
    return np.mean(np.sum((u - pre)**2, axis=0)**.5)
# end

def maxdist(u,pre):
    """ Max over i of ||u_i - pre_i||"""
    return np.max(np.sum((u - pre)**2, axis=0)**.5)
# end

def load_spec(i):
    return np.genfromtxt(FNAME.format(i), delimiter=",")
# end

def config_model(A):
    """Produce a network with the same degree distribution as the 
        network with adj matrix A"""

    # Get in and out deg distributions
    G    = nx.DiGraph(A)
    din  = sorted([ d for n,d in G.in_degree()], reverse=True)
    dout = sorted([d for n,d in G.out_degree()], reverse=True)
    
    # Create new adj matrix
    M    = nx.directed_configuration_model(din,dout, create_using=nx.DiGraph)
    M    = nx.DiGraph(M) # Removes parellel edges
    M    = nx.adj_matrix(M).toarray()
    
    return M
# end

def score_nodes(rc, u, t, r_0=None, u_0=None):
    """ Give every node in the reservoir a relative importance score
        
        Parameters
        ----------
        rc (ResComp): reservoir computer
        u  (solve_ivp solution): system to model
        t  (ndarray): time values to test 
        
        Returns
        -------
        scores (ndarray): Each node's importance score
    """
    pre, r     = rc.predict(t, return_states=True, r_0=r_0, u_0=u_0)
    derivative = rc.W_out.T.dot(pre - u(t))
    scores     = np.mean(np.abs(derivative*r), axis=1)
    return scores
# end

def avg_score_nodes(A, params, u, t, trials=10):
    """ Determines the most useful nodes in a network by averaging scores
        over a number of trials
    """
    scores = np.zeros(A.shape[0])
    for i in range(trials):
        rc = make_res_comp(A,params)
        rc.fit(t,u)
        scores += score_nodes(rc, u, t)
    
    return scores/trials
# end

def specialize_best_nodes(rc, how_many, u, t, r_0=None, u_0=None): 
    """ Specializes the most useful nodes in the reservoir and
        returns an adjacency matrix of the specialized reservoir
        
        Parameters
        ----------
        rc (ResComp): reservoir computer
        how_many (int): How many nodes to specialize
        u  (solve_ivp solution): system to model
        t  (ndarray): time values to test
        
        Returns
        S (ndarray): adj matrix of the specialized reservoir
    """
    scores     = score_nodes(rc, u, t, r_0=r_0, u_0=u_0)
    tot        = rc.res.shape[0]
    worst_idxs = list(np.argsort(scores)[1:(tot-how_many)])
    A          = rc.res
    A          = (A != 0)*1
    S          = specializeGraph(A, worst_idxs)
    return S
# end   

def spec_avg_best_nodes(rc, how_many, u, t, r_0=None, u_0=None, trials=1): 
    """ Specializes the most useful nodes on average and
        returns an adjacency matrix of the specialized reservoir
        
        Parameters
        ----------
        rc (ResComp): reservoir computer
        how_many (int): How many nodes to specialize
        u  (solve_ivp solution): system to model
        t  (ndarray): time values to test
        
        Returns
        S (ndarray): adj matrix of the specialized reservoir
    """
    scores     = avg_score_nodes(rc.res, rc.params, u, t, trials=trials, r_0=r_0, u_0=u_0)
    tot        = rc.res.shape[0]
    worst_idxs = list(np.argsort(scores)[1:(tot-how_many)])
    A          = rc.res
    A          = (A != 0)*1
    S          = specializeGraph(A, worst_idxs)
    return S
# end
    
def how_long_accurate(u, pre, tol=1):
    """ Find the first i such that ||u_i - pre_i||_2 > tol """
    for i in range(u.shape[1]):
        dist = np.sum((u[:,i] - pre[:,i])**2)**.5
        if dist > tol:
            return i        
    return u.shape[1]
# end

def make_res_comp(A, params):
    # Make res comp with desired adj matrix A
    new_params = copy.deepcopy(params)
    m,n    = A.shape    
    new_params["res_sz"] = n
    rc     = ResComp(3,3, **new_params)
    rc.res = A * new_params["spect_rad"] / max(np.abs(np.linalg.eigvals(A)))
    return rc
# end
