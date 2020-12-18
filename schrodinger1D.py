import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse import linalg as sla

def schrodinger1D(xmin, xmax, Nx, Vfun, params, neigs=20, findpsi=False):
    """
    Solves the 1 dimensional Schrodinger equation numerically
    
    Inputs
    ------
    xmin: float
        minimum value of the x axis
    xmax: float
        maximum value of the x axis
    Nx: int
        number of finite elements in the x axis
    Vfun: function
        potential energy function
    params: list
        list containing the parameters of Vfun
    neigs: int
        number of eigenvalues to find
    findpsi: bool
        If True, the eigen wavefunctions will be calculated and returned.
        If False, only the eigen energies will be found.
    
    Returns
    -------
    evl: np.array
        eigenvalues
    evt: np.array
        eigenvectors
    x: np.array
        x axis values
    """
    # for this code we are using Dirichlet Boundary Conditions
    x = np.linspace(xmin, xmax, Nx)  # x axis grid
    dx = x[1] - x[0]  # x axis step size
    # Obtain the potential function values:
    V = Vfun(x, params)
    # create the Hamiltonian Operator matrix:
    H = sparse.eye(Nx, Nx, format='lil') * 2
    for i in range(Nx - 1):
        #H[i, i] = 2
        H[i, i + 1] = -1
        H[i + 1, i] = -1
    #H[-1, -1] = 2
    H = H / (dx ** 2)
    # Add the potential into the Hamiltonian
    for i in range(Nx):
        H[i, i] = H[i, i] + V[i]
    # convert to csc matrix format
    H = H.tocsc()
    
    # obtain neigs solutions from the sparse matrix
    [evl, evt] = sla.eigs(H, k=neigs, which='SM')

    for i in range(neigs):
        # normalize the eigen vectors
        evt[:, i] = evt[:, i] / np.sqrt(
                                np.trapz(np.conj(
                                evt[:, i]) * evt[:, i], x))
        # eigen values MUST be real:
        evl = np.real(evl)
    if findpsi == False:
        return evl
    else: 
        return evl, evt, x

def eval_wavefunctions(xmin,xmax,Nx,Vfun,params,neigs,findpsi=True):
    """
    Evaluates the wavefunctions given a particular potential energy function Vfun
    
    Inputs
    ------
    xmin: float
        minimum value of the x axis
    xmax: float
        maximum value of the x axis
    Nx: int
        number of finite elements in the x axis
    Vfun: function
        potential energy function
    params: list
        list containing the parameters of Vfun
    neigs: int
        number of eigenvalues to find
    findpsi: bool
        If True, the eigen wavefunctions will be calculated and returned.
        If False, only the eigen energies will be found.
    """
    H = schrodinger1D(xmin, xmax, Nx, Vfun, params, neigs, findpsi)
    evl = H[0] # energy eigen values
    indices = np.argsort(evl)
    print("Energy eigenvalues:")
    for i,j in enumerate(evl[indices]):
        print("{}: {:.2f}".format(i+1,j))
    evt = H[1] # eigen vectors 
    x = H[2] # x dimensions 
    i = 0
    plt.figure(figsize=(8,8))
    while i < neigs:
        n = indices[i]
        y = np.real(np.conj(evt[:, n]) * evt[:, n])  
        plt.subplot(neigs, 1, i+1)  
        plt.plot(x, y)
        plt.axis('off')
        i = i + 1  
    plt.show()

def sho_wavefunctions_plot(xmin=-10, xmax=10, Nx=500, neigs=20, params=[1]):
    """
    Plots the 1D quantum harmonic oscillator wavefunctions.
    
    Inputs
    ------
    xmin: float
        minimum value of the x axis
    xmax: float
        maximum value of the x axis
    Nx: int
        number of finite elements in the x axis
    neigs: int
        number of eigenvalues to find          
    params: list
        list containing the parameters of Vfun     
    """
    def Vfun(x, params):
        V = params[0] * x**2
        return V
    
    eval_wavefunctions(xmin,xmax,Nx,Vfun,params,neigs,True)
    
def infinite_well_wavefunctions_plot(xmin=-10, xmax=10, Nx=500, neigs=20, params=1e10):
    """
    Plots the 1D infinite well wavefunctions.
    
    Inputs
    ------
    xmin: float
        minimum value of the x axis
    xmax: float
        maximum value of the x axis
    Nx: int
        number of finite elements in the x axis
    neigs: int
        number of eigenvalues to find          
    params: float
        parameter of Vfun     
    """
    def Vfun(x, params):
        V = x*0
        V[:100]=params
        V[-100:]=params
        return V
    
    eval_wavefunctions(xmin,xmax,Nx,Vfun,params,neigs,True)
    
def double_well_wavefunctions_plot(xmin=-10, xmax=10, Nx=500, neigs=20, params=[-0.5,0.01]):
    """
    Plots the 1D double well wavefunctions.
    
    Inputs
    ------
    xmin: float
        minimum value of the x axis
    xmax: float
        maximum value of the x axis
    Nx: int
        number of finite elements in the x axis
    neigs: int
        number of eigenvalues to find          
    params: list
        list of parameters of Vfun     
    """

    def Vfun(x, params):
        A = params[0]
        B = params[1]
        V = A * x ** 2 + B * x ** 4
        return V

    eval_wavefunctions(xmin,xmax,Nx,Vfun,params,neigs,True)
