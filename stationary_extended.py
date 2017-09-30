
import numpy as np
from GPy.kern.src.stationary import Stationary
from GPy.kern.src.stationary import ExpQuad, Matern32, Matern52


class StationaryExtended(Stationary):
    """
    Stationary kernels (covariance functions).

    Extends the stationary kernel with additional methods to faciliate for
    derivative observations. The approach used here is similar to that employed
    in the gradient observations kernel in GPflow 
    (https://github.com/GPflow/GPflow/tree/fast-grad) in that the derivatives
    are chained through tau=dr**2.
    See also this note by Rasmus Bonnevie:
    https://github.com/GPflow/GPflow/files/741860/diffkerns.pdf
    """

    def __init__(self, *pargs, **kwarg):
        super().__init__(*pargs, **kwarg)
    
    def dK_dt(self,t,n):
        raise NotImplementedError("implement nth derivative of covariance wrt tau=r**2 to use this method")
    
    def dK_dt_via_X(self, X, X2, n):
        """Compute the nth derivative of K wrt tau=r**2 going through X"""
        return self.dK_dt(self._scaled_dist(X, X2)**2, n)
 
    def dr_dx(self, X, X2, diff_x=(), diff_y=()):
        """
        Compute derivative of r wrt dimensions `diff_x` of 1st argument and
        `diff_y` of 2nd argument.

        Parameters:
        - diff_x'/'diff_y': Tuples of integers. Dimensions to differentiate
        with respect to in X and X2 respeectively.

        Return:
        - Appropriate derivative as np array [NxM]
        """
        shape = (X.shape[0], X2.shape[0])
        x_order, y_order = (len(diff_x), len(diff_y))
        diff_order = x_order + y_order
        dx = diff_x + diff_y
        invdist = self._inv_dist(X, X2)
        if X2 is None:
            invdist = invdist + invdist.T
            X2 = X
        assert diff_order <= 2
        if diff_order == 0:
            grad = self._scaled_dist(X, X2)
        elif diff_order == 1:
            d = (X[:,dx[0]][:,None] - X2[:,dx[0]][None,:])
            grad = invdist*d/self.lengthscale**2
            if y_order > 0: grad*=-1
        elif diff_order == 2:
            i, j = dx
            if i == j:
                grad = np.zeros(shape)
            else:
                grad = np.zeros(shape)
        return grad    

    def dt_dx(self, X, X2, diff_x=(), diff_y=()):
        """
        Compute derivative of tau=r**2 wrt dimensions `diff_x` of 1st argument 
        and `diff_y` of 2nd argument.

        Parameters:
        - diff_x'/'diff_y': Tuples of integers. Dimensions to differentiate
        with respect to in X and X2 respeectively.

        Return:
        - Appropriate derivative as np array [NxM]
        """
        shape = (X.shape[0], X2.shape[0])
        x_order, y_order = (len(diff_x), len(diff_y))
        diff_order = x_order + y_order
        dx = diff_x + diff_y
        assert diff_order <= 2
        if diff_order == 0:
            grad = self._scaled_dist(X, X2)**2
        elif diff_order == 1:
            d = (X[:,dx[0]][:,None] - X2[:,dx[0]][None,:])
            grad = 2.*d/self.lengthscale**2
            if y_order > 0:
                grad*=-1
        elif diff_order == 2:
            i, j = dx
            if i == j:
                grad = 2*np.ones(shape)/self.lengthscale**2
                if x_order == y_order: grad*=-1
            else:
                grad = np.zeros(shape)
        return grad
        
    def kernel_derivatives(self, X, X2=None, diff_x=(), diff_y=()):
        """
        Compute derivative of K wrt dimensions `diff_x` of 1st argument and
        `diff_y` of 2nd argument.

        Parameters:
        - diff_x'/'diff_y': Tuples of integers. Dimensions to differentiate
        with respect to in X and X2 respeectively.

        Return:
        - Appropriate derivative as np array [NxM]
        """
        x_order, y_order = (len(diff_x), len(diff_y))
        diff_order = x_order + y_order
        assert diff_order <= 2 # only up to 2nd deriv required for deriv obs
        dx = diff_x + diff_y
        
        if diff_order == 0:
            return self.K(X, X2)
        elif diff_order == 1:
            invdist = self._inv_dist(X, X2)
            dK_dr = self.dK_dr_via_X(X, X2)
            tmp = invdist*dK_dr
            if X2 is None:
                tmp = tmp + tmp.T
                X2 = X
            grad = tmp*(X[:,dx[0]][:,None] - X2[:,dx[0]][None,:])/self.lengthscale**2
            if y_order>0: grad *= -1
            # test
#            d = X[:,dx[0]][:,None]-X2[:,dx[0]][None,:] # equivalent to X-X2.T in 1d
#            grad2 = -d*self.K(X,X2)
#            if y_order>0: grad2 *= -1
#            print(np.sum(np.abs(grad2-grad)))
            # test2
#            grad2 = self.dK_dt_via_X(X,X2,diff_order)*self.dt_dx(X,X2,diff_x,diff_y)
#            grad2 = self.dK_dr_via_X(X,X2)*self.dr_dx(X,X2,diff_x,diff_y)
#            print(np.sum(np.abs(grad2-grad)))
            return grad
        elif diff_order == 2:
#            # Use existing gradients_XX function with dL_dK=1, then pick out required dims
            grad = self.gradients_XX(1, X, X2)[:,:,diff_x[0],diff_y[0]]
#            # Or use intuitive calculations (note doesn't handle inf*0 issues on diagonal for certain kernels):
#            grad2 = self.dK_dt_via_X(X,X2,1)*self.dt_dx(X,X2,diff_x,diff_y) + \
#                    self.dK_dt_via_X(X,X2,2)*self.dt_dx(X,X2,(),diff_y)*self.dt_dx(X,X2,diff_x,())
#            print(diff_x)
#            print(diff_y)
#            grad2 = self.dK_dr_via_X(X,X2)*self.dr_dx(X,X2,diff_x,diff_y) + \
#                    self.dK2_drdr_via_X(X,X2)*self.dr_dx(X,X2,(),diff_y)*self.dr_dx(X,X2,diff_x,())
#            print(np.isnan(grad2).sum())
#            print(np.sum(np.abs(grad2-grad)))
            return grad


class Matern32Extended(Matern32, StationaryExtended):
    """
    Extends the Matern32 kernel with methods needed for derivative observations
    """
    def __init__(self, *kargs, **pwargs):
        super().__init__(*kargs, **pwargs)

    def dK2_drdr(self, r):
        """
        ..math::
            r = \sqrt( \sum_{q=1}^Q (x_q - x'q)^2/l_q^2 )
        """
        return 3*self.variance * (np.sqrt(3)*r-1) * np.exp(-np.sqrt(3)*r)
    
    def dK_dt(self,t,n):
        K = self.K_of_r(np.sqrt(t))
        if n == 0:
            return K
        elif n == 1:
            return (-3/(2*np.sqrt(3*t) + 2))*K
        elif n == 2:
            return (3*np.sqrt(3)/(4*(np.sqrt(t) + np.sqrt(3)*t)))*K
        


class Matern52Extended(Matern52, StationaryExtended):
    """
    Extends the Matern52 kernel with methods needed for derivative observations
    """
    def __init__(self, *kargs, **pwargs):
        super().__init__(*kargs, **pwargs)

    def dK2_drdr(self, r):
        return 5./3*self.variance*(5.*r**2 -np.sqrt(5.)*r -1)*np.exp(-np.sqrt(5.)*r)
    
    def dK_dt(self,t,n):
        K = self.K_of_r(np.sqrt(t))
        if n == 0:
            return K
        elif n == 1:
            return (-5/3*(np.sqrt(t) + np.sqrt(5)*t) / (10/3*t**(3/2) + 2*np.sqrt(t) + 2*np.sqrt(5)*t))* K
        elif n == 2:
            return ( (25/3*t**(5/2)) / (20/3*t**(7/2) + 4*t**(5/2) + 4*np.sqrt(5)*t**3)) * K


class ExpQuadExtended(ExpQuad, StationaryExtended):
    """
    Extends the ExpQuad kernel with methods needed for derivative observations
    """
    def __init__(self, *pargs, **kwargs):
        super().__init__(*pargs, **kwargs)
        
    def dK2_drdr(self, r):
        return (r**2-1)*self.K_of_r(r)

    def dK_dt(self,t,n):
        if n==0:
            return self.variance*np.exp(-t)
        else:
            return self.variance*(-0.5)**n*np.exp(-0.5*t)

        
if __name__ == "__main__":
    pass
