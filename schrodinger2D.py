import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse import linalg as sla
from mpl_toolkits.mplot3d import Axes3D
 

def schrodinger2D(xmin, xmax, Nx, ymin, ymax, Ny, Vfun2D, params, neigs, E0 = 0.0, findpsi = False):
    """
    Solves the 2 dimensional Schrodinger equation numerically.
    
    Inputs
    ------
    xmin: float
        minimum value of the x axis
    xmax: float
        maximum value of the x axis
    Nx: int
        number of finite elements in the x axis
    ymin: float
        minimum value of the y axis
    ymax: float
        maximum value of the y axis
    Ny: int
        number of finite elements in the y axis        
    Vfun2D: function
        potential energy function
    params: list
        list containing the parameters of Vfun
    neigs: int
        number of eigenvalues to find
    E0: float
        eigenenergy value to solve for
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
    y: np.array
        y axis values
    """
    x = np.linspace(xmin, xmax, Nx)  
    dx = x[1] - x[0]  
    y = np.linspace(ymin, ymax, Ny)
    dy = y[1] - y[0]

    V = Vfun2D(x, y, params)

    # The following code might be problematic.
    '''
    # create the 2D Hamiltonian matrix
    Hx = create_1d_hamiltonian(Nx, dx)
    Hy = create_1d_hamiltonian(Ny, dy)
    
    Ix = sparse.eye(Nx, Nx)
    Iy = sparse.eye(Ny, Ny)
    H = sparse.kron(Hx, Iy) + sparse.kron(Ix, Hy)  
    '''
    # Use the following instead of the above!
    H = create_2d_hamiltonian(Nx, dx, Ny, dy)

    # Convert to lil form and add potential energy function
    H = H.tolil()
    for i in range(Nx * Ny):
        H[i, i] = H[i, i] + V[i]    

    # convert to csc form and solve the eigenvalue problem
    H = H.tocsc()  
    [evl, evt] = sla.eigs(H, k = neigs, sigma = E0)
            
    if findpsi == False:
        return evl
    else: 
        return evl, evt, x, y


def create_1d_hamiltonian(Nx, dx):
    """
    Creates a 1 dimensional Hamiltonian.
   
    Inputs
    ------
    Nx: int
        number of elements in that axis
    dx: float
        step size
       
    Returns
    -------
    H: np.array
        np.array of the Hamiltonian
    """
    H = sparse.eye(Nx, Nx, format = "lil") * 2
    for i in range(Nx - 1):
        H[i, i + 1] = -1
        H[i + 1, i] = -1
    H = H / (dx ** 2)  
    return H


def create_2d_hamiltonian(Nx, dx, Ny = None, dy = None):
    """
    Creates a 2D Hamiltonian matrix.

    Inputs
    ------
    Nx, Ny: int
        Number of elements in that axis.
    dx, dy: float
        Step size.

    Returns
    -------
    H: np.array
        Hamiltonian matrix.
    """
    if Ny is None:
        Ny = Nx
    if dy is None:
        dy = dx
    Hx = create_1d_hamiltonian(Nx, dx)
    Iy = sparse.eye(Ny, Ny)
    Hy = create_1d_hamiltonian(Nx * Ny, dy)
    H = sparse.kron(Hx, Iy) + Hy
    return H

    
def eval_wavefunctions(xmin, xmax, Nx,
                       ymin, ymax, Ny,
                       Vfun, params, neigs, E0, findpsi):
 
    """
    Evaluates and plots the 2 dimensional Schrodinger equation numerically for some potential function Vfun.
    The 2D wavefunctions (actually, the probabilities!) are plotted as a heatmap instead of as an actual plot.
    
    Inputs
    ------
    xmin: float
        minimum value of the x axis
    xmax: float
        maximum value of the x axis
    Nx: int
        number of finite elements in the x axis
    ymin: float
        minimum value of the y axis
    ymax: float
        maximum value of the y axis
    Ny: int
        number of finite elements in the y axis        
    Vfun: function
        potential energy function
    params: list
        list containing the parameters of Vfun
    neigs: int
        number of eigenvalues to find
    E0: float
        eigenenergy value to solve for
    findpsi: bool
        If True, the eigen wavefunctions will be calculated and returned.
        If False, only the eigen energies will be found.
    """ 
        
    H = schrodinger2D(xmin, xmax, Nx, ymin, ymax, Ny, Vfun, params, neigs, E0, findpsi)
    evl = H[0] # eigenvalues
    indices = np.argsort(evl)
    print("Energy eigenvalues:")
    for i,j in enumerate(evl[indices]):
        print("{}: {:.2f}".format(i + 1, np.real(j)))
        
    evt = H[1] # eigenvectors
    
    plt.figure(figsize = (15, 15))
    # unpack the vector into 2 dimensions for plotting:
    for n in range(neigs):
        psi = evt[:, n]  
        PSI = psi.reshape(Nx, Ny) 
        PSI = np.abs(PSI) ** 2
        plt.subplot(2, int(neigs / 2), n + 1)    
        plt.pcolormesh(np.transpose(PSI), cmap = "jet")
        plt.axis("equal")
        plt.axis("off")
    plt.show()


def sho_wavefunctions_plot(xmin = -10, xmax = 10, Nx = 250,
                           ymin = -10, ymax = 10, Ny = 250,
                           params = [1, 1], neigs = 6, E0 = 0, findpsi = True):
    """
    Evaluates and plots the 2D QSHO wavefunctions.
    
    Inputs
    ------
    xmin: float
        minimum value of the x axis
    xmax: float
        maximum value of the x axis
    Nx: int
        number of finite elements in the x axis
    ymin: float
        minimum value of the y axis
    ymax: float
        maximum value of the y axis
    Ny: int
        number of finite elements in the y axis        
    params: list
        list containing the parameters of Vfun
    neigs: int
        number of eigenvalues to find
    E0: float
        eigenenergy value to solve for
    findpsi: bool
        If True, the eigen wavefunctions will be calculated and returned.
        If False, only the eigen energies will be found.    
    """
    
    def Vfun(X, Y, params):
        Nx = len(X)
        Ny = len(Y)
        M = Nx * Ny
        V = np.zeros(M)
        vindex = 0
        for i in range(Nx):
            for j in range(Ny):
                V[vindex] = params[0] * X[i]**2 + params[1] * Y[j]**2
                vindex = vindex + 1
        return V
    
    eval_wavefunctions(xmin, xmax, Nx,
                       ymin, ymax, Ny,
                       Vfun, params, neigs, E0, findpsi)


def stadium_wavefunctions_plot(R = 1, L = 2, V0 = 1e6, neigs = 6, E0 = 500, Ny = 250):
    """
    Evaluates and plots the 2D stadium potential wavefunctions.
    
    Inputs
    ------
    R: float
        stadium radius
    L: float
        stadium length
    V0: float
        stadium wall potential
    neigs: int
        number of eigenvalues to solve for
    E0: float
        eigenvalue to solve for
    Ny: int
        number of elements in the y axis
    """
    ymin = -0.5 * L - R
    ymax = 0.5 * L + R
    xmin = -R
    xmax = R
    params = [R, L, V0]
    print("xmin: {}, xmax: {}, ymin: {}, ymax: {}".format(xmin, xmax, ymin, ymax))

    Nx = int(Ny * 2 * R / (2.0 * R + L))
    print("Nx: {}, Ny: {}".format(Nx, Ny))
    
    def Vfun2D(X, Y, params):
        R = params[0] # stadium radius
        L = params[1] # stadium length
        V0 = params[2] # stadium wall potential
        # stadium potential function
        Nx = len(X)
        Ny = len(Y)
        [x, y] = np.meshgrid(X, Y)
        F = np.zeros([Nx, Ny])

        for i in range(Nx):
            for j in range(Ny):
                if abs(X[i]) == R or abs(Y[j]) == R + 0.5 * L:
                    F[i, j] = V0
                if (abs(Y[j]) - 0.5 * L) > 0 and np.sqrt((abs(Y[j]) - 0.5 * L) ** 2 + X[i] ** 2) >= R:
                    F[i, j] = V0
                    
        V = F.reshape(Nx * Ny)             
        return V
 
    eval_wavefunctions(xmin, xmax, Nx,
                       ymin, ymax, Ny,
                       Vfun2D, params, neigs, E0, findpsi=True)


def stadium_wavefunctions_3dplot(R = 1, L = 0, V0 = 1e6, neigs = 6, E0 = 70, Ny = 250):
    """
    Evaluates and plots the 2D stadium potential wavefunctions.
    Instead of plotting the wavefunctions as a heatmap, plot them as a 3D surface instead
    
    Inputs
    ------
    R: float
        stadium radius
    L: float
        stadium length
    V0: float
        stadium wall potential
    neigs: int
        number of eigenvalues to solve for
    E0: float
        eigenvalue to solve for
    Ny: int
        number of elements in the y axis
    """ 
    ymin = -0.5 * L - R
    ymax = 0.5 * L + R
    xmin = -R
    xmax = R
    params = [R, L, V0]
    print("xmin: {}, xmax: {}, ymin: {}, ymax: {}".format(xmin, xmax, ymin, ymax))

    Nx = int(Ny * 2 * R / (2.0 * R + L))
    print("Nx: {}, Ny: {}".format(Nx, Ny))
    
    def Vfun2D(X, Y, params):
        R = params[0] # stadium radius
        L = params[1] # stadium length
        V0 = params[2] # stadium wall potential
        # stadium potential function
        Nx = len(X)
        Ny = len(Y)
        [x, y] = np.meshgrid(X, Y)
        F = np.zeros([Nx, Ny])

        for i in range(Nx):
            for j in range(Ny):
                if abs(X[i]) == R or abs(Y[j]) == R + 0.5 * L:
                    F[i, j] = V0
                if (abs(Y[j]) - 0.5 * L) > 0 and np.sqrt((abs(Y[j]) - 0.5 * L) ** 2 + X[i] ** 2) >= R:
                    F[i, j] = V0
                    
        V = F.reshape(Nx * Ny)             
        return V
    
    H = schrodinger2D(xmin, xmax, Nx, ymin, ymax, Ny, Vfun2D, params, neigs, E0, True)
    evl = H[0] # eigenvalues
    indices = np.argsort(evl)
    print("Energy eigenvalues:")
    for i,j in enumerate(evl[indices]):
        print("{}: {:.2f}".format(i + 1, np.real(j)))
    evt = H[1] # eigenvectors

    # unpack the vector into 2 dimensions for plotting:
    for n in range(evt.shape[1]):
        psi = evt[:, n]  
        PSI = psi.reshape(Nx, Ny) 
        PSI = np.abs(PSI)**2
        fig = plt.figure(figsize = (10, 10))
        ax = fig.add_subplot(111, projection = "3d")
        X, Y = np.meshgrid(H[2], H[3])
        ax.plot_surface(X, Y , np.transpose(PSI), cmap = "jet")
        ax.axis("off")
        plt.show()    