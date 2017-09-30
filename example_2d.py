#%% A simple 2d example

#%% Imports
import numpy as np
import matplotlib.pyplot as plt
import GPy
from GPy.util.multioutput import build_XY
from gp_derivative_observations import GPDerivativeObservations


#%%
plt.close('all')

#%% Options
deriv_dim = 1 # 0 or 1


#%%
seed=1
np.random.seed(seed=seed)

#%% Generate data
# sample inputs and outputs
plot_lims = [0,7]
nX = 20
X = np.random.uniform(0,7,(nX,2))
R = np.sqrt(np.sum(X**2, axis=1)).reshape(-1,1)
c1 = 1; c2 = 0.4
Y0 = c2*X[:,0,None] + np.sin(c1*X[:,1,None]) + np.random.randn(nX,1)*0.05   # y
Y12 = c1*np.cos(c1*X[:,1,None]) + np.random.randn(nX,1)*0.05           # dy_dx1
Y11 = c2 + np.random.randn(nX,1)*0.05                                  # dy_dx2

#%% Fit a simple GP model
ker = GPy.kern.ExpQuad(2)
m_simple = GPy.models.GPRegression(X,Y0,ker)
m_simple.optimize(max_f_eval = 1000)
print(m_simple)
ax = m_simple.plot()
ax.set_xlabel('x1')
ax.set_ylabel('x2')

#%% Fit derivative GP
index=[0,1,2]
m = GPDerivativeObservations(X_list=[X,X,X], Y_list=[Y0,Y11,Y12], index=index)
m.optimize('bfgs', max_iters=100)
print(m)

#%% 1D plot slices
slices = [0, 4]
fig, ax = plt.subplots(1, len(slices), figsize=(8,3))
nXs=200                                         # dense sampling for plotting
for i, y in zip(range(len(slices)), slices):
    Xs = np.linspace(*plot_lims,nXs)
    Xs = (np.concatenate((Xs,Xs))).reshape(-1,2)
    Xs[:,1-deriv_dim] = slices[i]*np.ones_like(Xs[:,0])
    Xs_, _, _ = build_XY([Xs,Xs,Xs],index=index)
    _,id_u = np.unique(Xs_[:,-1,None],return_inverse=True)
    output_index = (id_u.min() + id_u).reshape(Xs_[:,-1,None].shape)
    
    # Truth
    Ys_true = c2*Xs[:,0,None] + np.sin(c1*Xs[:,1,None])
    ax[i].plot(Xs[:int(nXs/2),deriv_dim], Ys_true[:int(nXs/2),0], 'k-', lw= 0.7,
                  label='Truth')
    ax[i].set_xlim(plot_lims)
    
    # Simple GP
    Ys, _ = m_simple.predict(Xs)
    ax[i].plot(Xs[:int(nXs/2),deriv_dim], Ys[:int(nXs/2),0],'--',color='C0', 
                  label='Pred (w/o derivs)')
    
    # Derivative observations GP
    Ys, _ = m.predict(Xs_, Y_metadata={'output_index':output_index})
    ax[i].plot(Xs[:int(nXs/2),deriv_dim], Ys[:int(nXs/2),0],'--',color='C3', 
                  label='Pred (with derivs)')
    
    # Title, axis labels
    ax[i].set_title('x1 = {}'.format(slices[i]))
    ax[i].set_xlabel('x2')
    ax[i].set_ylabel('f(x1,x2)')

ax[0].legend(loc='lower left')
fig.tight_layout()
    











