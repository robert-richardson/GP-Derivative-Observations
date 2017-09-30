
from GPy.kern.src.kern import Kern
import numpy as np
from stationary_extended import ExpQuadExtended

class DerivativeObservations(Kern):
    """
    Covariance function for derivative observations models
    """
    def __init__(self, input_dim, output_dim, active_dims=None, name='deriv_obs', base_kernel=None):
#        assert input_dim == 2, "For the moment assume input_dim=2 (1 dim + 1 index)"
        super().__init__(input_dim, active_dims, name)
        self.output_dim = output_dim
        if base_kernel is None:
            base_kernel = ExpQuadExtended(input_dim-1)
        self.base_kernel = base_kernel
        self.link_parameters(self.base_kernel)
    
    def parameters_changed(self):
        pass

    def K(self, X, X2=None):
        if X2 is None: X2=X
        index_1 = np.unique(X[:,-1]).astype(int)
        index_2 = np.unique(X2[:,-1]).astype(int)
        #loop over all cases to iteratively construct block matrix
        rows = []
        for i in index_1:
            row_i = []
            for j in index_2:
                idx, idx2 = (X[:,-1]==i, X2[:,-1]==j)
                diff_x, diff_y = (self.ind2dx(i),self.ind2dx(j))
                row_i.append(self.base_kernel.kernel_derivatives(X[idx,:-1],
                                                  X2[idx2,:-1],
                                                  diff_x, diff_y))
            rows.append(np.concatenate(row_i, 1))
        K = np.concatenate(rows, 0)
        return K
    
    def Kdiag(self, X):
        return np.diag(self.K(X))
    
    def ind2dx(self, index):
        '''Convert a linear index into a tuple of integers to differentiate with respect to.'''
        return () if index == 0 else (index-1,)
    

    def update_gradients_full(self, dL_dK, X, X2=None):
        """
        Given the derivative of the objective wrt the covariance matrix
        (dL_dK), compute the gradient wrt the parameters of the full kernel
        by summing the contributions from each subkernel where:
        K_full = [K, dKdx; dKdxp, d2Kdx2]
        """
        if X2 is None: X2 = X
        index_1 = np.unique(X[:,-1]).astype(int)
        index_2 = np.unique(X2[:,-1]).astype(int)
        k = self.base_kernel
        k.variance.gradient = np.sum(self.K(X, X2)*dL_dK)/k.variance # variance subkernels are all equivalent
        k.lengthscale.gradient = 0
        for i in index_1:
            for j in index_2:
                idx, idx2 = (X[:,-1]==i, X2[:,-1]==j)
                diff_x, diff_y = (self.ind2dx(i),self.ind2dx(j))
                k.lengthscale.gradient += self._lengthscale_deriv_grads( \
                        dL_dK[np.where(np.outer(idx,idx2))].reshape(np.sum(idx),np.sum(idx2)), \
                        X[idx,:-1], X2[idx2,:-1], diff_x, diff_y)

    def _lengthscale_deriv_grads(self, dL_dK, X, X2=None, diff_x=(), diff_y=()):
        """
        Given the derivative of the objective wrt the covariance matrix, dL_dK,
        compute the gradient wrt the lengthscale (l) of the 0th, 1st or 2nd
        derivative of the kernel:
        dL_dl = dL_dK*{dK_dl, d(dKdx)_dl, d(dKdxp)_dl, d(d2Kdx2)_dl}
        
        Parameters:
        - diff_x'/'diff_y': Tuples of integers. Dimensions to differentiate
        with respect to in X and X2 respeectively.

        Return:
        - Appropriate gradient
        """
        x_order, y_order = (len(diff_x), len(diff_y))
        diff_order = x_order + y_order
        assert diff_order <= 2 # only up to 2nd deriv required for deriv obs
#        dx = diff_x + diff_y
        k = self.base_kernel
        L = k.lengthscale
        r = k._scaled_dist(X, X2)            
        if diff_order == 0: # dK_dl
#            dK_dl_1d = (r**2/L) * k.K(X, X2)
#            # Multi-dimensional (1d is equivalent to multi dim in this case)
            dK_dl = -(r/L)*k.dK_dr_via_X(X, X2)
        elif diff_order == 1: # d(dK_dx)_dl
            d2t_dldx = -2*k.dt_dx(X,X2,diff_x,diff_y)/L
            dt_dl = -2*r**2/L
            dK_dl = k.dK_dt_via_X(X,X2,1)*d2t_dldx + \
                        k.dK_dt_via_X(X,X2,2)*k.dt_dx(X,X2,diff_x,diff_y)*dt_dl
            # Test against  intuitive implementation in 1d:
#            d = (X[:,dx[0]][:,None] - X2[:,dx[0]][None,:])/L
#            dK_dl_1d = d*(2 - r**2)/(L**2) * k.K(X, X2)
#            if y_order>0: dK_dl_1d*=-1
#            print(np.sum(np.abs(dK_dl-dK_dl_1d)))
        elif diff_order == 2: # d(d2K_dx2)_dl
            # Multi-dimensional (correct)
            dt_dl = -2*r**2/L
            dt_dx = k.dt_dx(X,X2,diff_x,())
            dt_dy = k.dt_dx(X,X2,(),diff_y)
            d2t_dldx = -2*k.dt_dx(X,X2,diff_x,())/L
            d2t_dldy = -2*k.dt_dx(X,X2,(),diff_y)/L
            d3t_dldxdy = -2*k.dt_dx(X,X2,diff_x,diff_y)/L
            dK_dl = k.dK_dt_via_X(X,X2,1)*d3t_dldxdy + \
                        k.dK_dt_via_X(X,X2,2) * ( dt_dl*k.dt_dx(X,X2,diff_x,diff_y) \
                                      + dt_dx*d2t_dldy + dt_dy*d2t_dldx ) + \
                        k.dK_dt_via_X(X,X2,3)*dt_dl*dt_dx*dt_dy
            # Test against intuitive implementation in 1d:
#            dK_dl_1d = (-r**4 + 5*r**2 - 2)/(L**3) * k.K(X, X2)
#            print(np.sum(np.abs(dK_dl_1d-dK_dl)))
        return np.sum(dK_dl*dL_dK)

        
if __name__ == "__main__":
    pass


        
    
    
    
    
