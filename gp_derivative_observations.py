

import numpy as np
from GPy.core import GP
from GPy import util
from derivative_observations import DerivativeObservations
from stationary_extended import ExpQuadExtended

class GPDerivativeObservations(GP):
    """
    Gaussian Process model for Derivative Observations

    This is a thin wrapper around the models.GP class, with a set of sensible defaults
    The format is very similar to that used for the coregionalisation class.

    :param X_list: list of input observations corresponding to each output
    :type X_list: list of numpy arrays
    :param Y_list: list of observed values related to the different noise models
    :type Y_list: list of numpy arrays
    :param kernel: a GPy Derivative Observations kernel (if previously defined)
    :param base_kernel: a GPy kernel from which to build a Derivative 
                        Observations kernel ** defaults to RBF **
    :type kernel: None | GPy.kernel defaults
    :param base_kernel: a GPy kernel ** defaults to RBF **
    :type base_kernel: None | GPy.kernel defaults
    :likelihoods_list: a list of likelihoods, defaults to list of Gaussian likelihoods
    :type likelihoods_list: None | a list GPy.likelihoods
    :param name: model name
    :type name: string
    :param kernel_name: name of the kernel
    :type kernel_name: string
    """
    def __init__(self, X_list, Y_list, index=None, kernel=None, likelihoods_list=None, base_kernel=None, name='GPDO', kernel_name='deriv_obs'):

        #Input and Output
        assert kernel is None or base_kernel is None
        Ny = len(Y_list)
        X,Y,self.output_index = util.multioutput.build_XY(X_list,Y_list,index)
        
        #Make output index unique ordered values
        _,id_u = np.unique(self.output_index,return_inverse=True)
        self.output_index = (id_u.min() + id_u).reshape(self.output_index.shape)

        #Kernel
        if kernel is None:
            if base_kernel is None:
                base_kernel = ExpQuadExtended(X.shape[1]-1)
            kernel = DerivativeObservations(X.shape[1], output_dim=Ny, active_dims=None, base_kernel=base_kernel)

        #Likelihood
        likelihood = util.multioutput.build_likelihood(Y_list,self.output_index,likelihoods_list)

        super().__init__(X,Y,kernel,likelihood, Y_metadata={'output_index':self.output_index})





