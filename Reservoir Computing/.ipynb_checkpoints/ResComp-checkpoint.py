import numpy as np
from sklearn.linear_model import Ridge
from scipy import integrate

class ResComp:
    def __init__(self, num_in, num_out, res_sz=200, activ_f=np.tanh, connect_p=.1, ridge_alpha=.00001,spect_rad=.9, gamma = 1.,sigma = 0.1, uniform_weights=False, solver="ridge regression"):
        
        # Set model attributes
        self.W_in        = np.random.rand(res_sz, num_in) - .5
        self.W_out       = np.zeros((num_out, res_sz))
        self.gamma       = gamma
        self.sigma       = sigma
        self.activ_f     = activ_f
        self.ridge_alpha = ridge_alpha
        self.state_0     = np.random.rand(res_sz)
        self.solver      = solver
        
        # Make reservoir
        if uniform_weights:
            self.res = 1.*(np.random.rand(res_sz, res_sz) < connect_p)
        else:
            self.res   = np.random.rand(res_sz, res_sz) - .5
            self.res[np.random.rand(res_sz,res_sz) > connect_p] = 0
            self.res *= spect_rad/max(np.linalg.eigvals(self.res)).real
        
    # end
    
    def drive(self,t,u):
        """
        Parameters
        t (1 dim ndarray): an array of time values
        u (function)     : for each i, u(t[i]) produces the state of the system that is being learned
        """
        
        # Reservoir ode
        def res_f(r,t):
            return self.gamma*(-1*r + self.activ_f( self.res.dot(r) + self.sigma*self.W_in.dot(u(t))))
        #end
        
        r_0    = self.state_0
        states = integrate.odeint(res_f,r_0,t)
        self.state_0 = states[-1]
        return states
    # end
    
    
    def fit(self, t, u):
        """
        Parameters
        t (1 dim ndarray): an array of time values
        u (function)     : for each i, u(t[i]) produces the state of the system that is being learned
        """
        
        driven_states    = self.drive(t,u)
        true_states      = u(t).T
        
        if self.solver == "least squares":
            self.W_out = np.linalg.lstsq(driven_states, true_states)[0].T
        # end
        else:
            ridge_regression = Ridge(alpha=self.ridge_alpha, fit_intercept=False)
            ridge_regression.fit(driven_states,true_states)
            self.W_out       = ridge_regression.coef_
            
        error = np.mean(np.linalg.norm(self.W_out.dot(driven_states.T)-true_states.T,ord=2,axis=0))
        return error
    # end
    
 
    def predict(self, t, u_0=None, r_0=None, return_states=False):
        # Reservoir prediction ode
        
        def res_pred_f(r,t):
            return self.gamma*(-1*r + self.activ_f( self.res.dot(r) + self.sigma * self.W_in.dot(self.W_out.dot(r))))
        # end
        
        if r_0 is None and u_0 is None:
            r_0  = self.state_0
        # end
        
        elif r_0 is None and u_0 is not None:
            r_0 = res.W_in.dot(u_0)

        pred = integrate.odeint(res_pred_f, r_0, t)
        
        if return_states:
            return self.W_out.dot(pred.T), pred.T
        return self.W_out.dot(pred.T)
    # end
    

