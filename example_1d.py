#%% A simple 1d example

#%% Imports
import numpy as np
import matplotlib.pyplot as plt
import GPy
from GPy.util.multioutput import build_XY
from gp_derivative_observations import GPDerivativeObservations

#%%
plt.close('all')

#%% Options
plot_simple = True      # Plot simple GP (no derivatives observed)
plot_deriv_obs = True   # Plot observed derivative data
plot_main = True        # Plot GPDerivativeObservations prediction of func.
plot_deriv_preds = True # Plot GPDerivativeObservations prediction of deriv.
nX0, nX1 = 15, 10       # no. of func. and deriv. observations respectively

#%%
seed=1
np.random.seed(seed=seed)

#%% Generate data

#test function and derivatives
c = 1
f = lambda x: np.sin(c*x)
fp = lambda x: c*np.cos(c*x)

#X0 = np.linspace(0, 7, nX0).reshape(-1,1); X1 = X0
#f = lambda x: 1.3*np.sin(x) + 0.5*np.cos(1.5*x) +x/6.
#fp = lambda x: 1.3*np.cos(x) -1.5*0.5*np.sin(1.5*x) + 1/6.

#build a design matrix with a column of integers indicating the output
X0 = np.random.uniform(0, 5, (nX0, 1))
X1 = np.random.uniform(3, 8, (nX1, 1))

#build a suitable set of observed variables
noise_error_1, noise_error_2 = (0.075, 0.075)
Y0 = f(X0) + np.random.randn(*X0.shape) * noise_error_1
Y1 = fp(X1) + np.random.randn(*X1.shape) * noise_error_2


#%% Plot truth

# Prepare
fig, ax = plt.subplots(figsize=(7,4))
fig.tight_layout()
col1, col2, col3 = ('C0','C3','C7')
plot_limits = [-3,11]
ax.hlines(0, plot_limits[0], plot_limits[1])

# Plot truth
nX0s = 200
nX1s = nX0s
X0s = np.linspace(plot_limits[0], plot_limits[1], nX0s)
X1s = np.linspace(plot_limits[0], plot_limits[1], nX1s)
Xs = np.concatenate((X0s,X1s))
Y0s = f(X0s); Y1s = fp(X1s)
ax.plot(X0s,Y0s,'-',lw=0.75,color=col1, label='f(x)')
ax.plot(X0,Y0,'x',label='f(x) (observed)',color=col1)
if plot_deriv_obs:
    ax.plot(X1s,Y1s,'-',lw=0.75,color=col2, label='fp(x)')
    ax.plot(X1,Y1,'x',label='fp(x) (observed)',color=col2)


#%% Simple GP
m_simple = GPy.models.GPRegression(X0, Y0, kernel=GPy.kern.ExpQuad(1,variance=5.01,lengthscale=5.02))
m_simple.optimize()
print(m_simple)


#%% Derivative GP
m = GPDerivativeObservations(X_list=[X0,X1], Y_list=[Y0,Y1])

#Note how this works analgously to the coregionalisation class
#m = GPy.models.GPCoregionalizedRegression(X_list=[X0,X1], Y_list=[Y0,Y1]) 

#%% Optimise
m.optimize('bfgs', max_iters=100)

#%% Plot GP

#Plot simple GP
if plot_simple:
    Ys_simple, Ys_var_simple = m_simple.predict(X0s.reshape(-1,1))
    Ys_err = 2*np.sqrt(Ys_var_simple)
    ax.plot(X0s.reshape(-1,1),Ys_simple[:nX0s,0],'-.',color=col3, label='f(x) (simple GP)')
    ax.fill_between(X0s,Ys_simple[:nX0s,0]-Ys_err[:nX0s,0], 
                    Ys_simple[:nX0s,0]+Ys_err[:nX0s,0],color=col3,alpha=0.15)


# Plot derivative observations GP
Xs_, _, _ = build_XY([X0s.reshape(-1,1),X1s.reshape(-1,1)])
Ys, Ys_var = m.predict(Xs_, Y_metadata={'output_index':Xs_[:,-1,None].astype(int)})
Ys_var[:nX0s,:] += m.mixed_noise.Gaussian_noise_0.variance.values
Ys_var[nX0s:,:] += m.mixed_noise.Gaussian_noise_1.variance.values
Ys_err = 2*np.sqrt(Ys_var)
if plot_main:
    ax.plot(X0s.reshape(-1,1),Ys[:nX0s,0],'--',color=col1, label='f(x) (deriv GP)')
    ax.fill_between(X0s,Ys[:nX0s,0]-Ys_err[:nX0s,0],Ys[:nX0s,0]+Ys_err[:nX0s,0],color=col1,alpha=0.15)
if plot_deriv_preds:
    ax.plot(X1s.reshape(-1,1),Ys[nX0s:,0],'--',color=col2,label='fp(x) (deriv GP)')
    ax.fill_between(X1s,Ys[nX0s:,0]-Ys_err[nX0s:,0],Ys[nX0s:,0]+Ys_err[nX0s:,0],color=col2,alpha=0.15)

# Legend etc.
ax.legend()

#%% Export pdf figure for github readme
#from matplotlib.backends.backend_pdf import PdfPages
#pp = PdfPages('coverfig.pdf')
#pp.savefig(fig)
#pp.close()

#%% Plot slices
#m_simple.plot()
fig2, ax2 = plt.subplots(figsize=(7,4)); fig2.tight_layout()
ax2.hlines(0, plot_limits[0], plot_limits[1])
slices = GPy.util.multioutput.get_slices([X0,X1])
m.plot(fixed_inputs=[(1,0)],which_data_rows=slices[0],Y_metadata={'output_index':0},plot_limits=plot_limits,ax=ax2)
m.plot(fixed_inputs=[(1,1)],which_data_rows=slices[1],Y_metadata={'output_index':1},plot_limits=plot_limits,ax=ax2)

#%% Test
print('\n', m)



