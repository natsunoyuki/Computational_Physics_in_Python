import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse, linalg
from scipy.sparse import linalg as sla
from mpl_toolkits.mplot3d import Axes3D
 
def schrodinger2D(xmin, xmax, Nx, ymin, ymax, Ny, Vfun2D, params, neigs, E0=0.0, findpsi=False):
    """
    Solves the 2 dimensional Schrodinger equation numerically
    
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

    # create the 2D Hamiltonian matrix
    Hx = create_hamiltonian(Nx, dx)
    Hy = create_hamiltonian(Ny, dy)

    Ix = sparse.eye(Nx, Nx)
    Iy = sparse.eye(Ny, Ny)
    H = sparse.kron(Iy, Hx) + sparse.kron(Hy, Ix)  

    # Convert to lil form and add potential energy function
    H = H.tolil()
    for i in range(Nx * Ny):
        H[i, i] = H[i, i] + V[i]    

    # convert to csc form and solve the eigenvalue problem
    H = H.tocsc()  
    [evl, evt] = sla.eigs(H, k=neigs, sigma=E0)
            
    if findpsi == False:
        return evl
    else: 
        return evl, evt, x, y

def create_hamiltonian(Nx, dx):
    """
    Creates a 1 dimensional Hamiltonian
   
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
    H = sparse.eye(Nx, Nx, format='lil') * 2
    for i in range(Nx - 1):
        H[i, i + 1] = -1
        H[i + 1, i] = -1
    H = H / (dx ** 2)  
    return H
    
def eval_wavefunctions(xmin, xmax, Nx,
                       ymin, ymax, Ny,
                       Vfun, params, neigs, E0, findpsi):
 
    """
    Evaluates and plots the 2 dimensional Schrodinger equation numerically for some potential function Vfun
    
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
        
    H = schrodinger2D(xmin,xmax,Nx,ymin,ymax,Ny,Vfun,params,neigs,E0,findpsi)
    evl = H[0] # eigenvalues
    indices = np.argsort(evl)
    print("Energy eigenvalues:")
    for i,j in enumerate(evl[indices]):
        print("{}: {:.2f}".format(i + 1, np.real(j)))
    evt = H[1] # eigenvectors
    plt.figure(figsize=(15, 15))
    # unpack the vector into 2 dimensions for plotting:
    for n in range(neigs):
        psi = evt[:, n]  
        PSI = oneD_to_twoD(Nx, Ny, psi)
        PSI = np.abs(PSI)
        plt.subplot(2, int(neigs/2), n + 1)    
        plt.pcolormesh(np.flipud(PSI), cmap = 'jet')
        plt.axis('equal')
        plt.axis('off')
    plt.show()

def twoD_to_oneD(Nx, Ny, F):
    """
    from a 2D matrix F return a 1D vector V
    
    Inputs
    ------
    Nx: int
        number of elements in the x axis
    Ny: int
        number of elements in the y axis
    F: np.array
        2D np.array to convert to 1D
        
    Returns
    -------
    V: np.array
        converted 1D array
    """
    V = np.zeros(Nx * Ny)
    vindex = 0
    for i in range(Ny):
        for j in range(Nx):
            V[vindex] = F[i, j]
            vindex = vindex + 1                      
    return V

def oneD_to_twoD(Nx, Ny, psi):
    """
    from a 1D vector psi return a 2D matrix PSI
    
    Inputs
    ------
    Nx: int
        number of elements in the x axis
    Ny: int
        number of elements in the y axis
    psi: np.array
        1D np.array to convert to 2D
        
    Returns
    -------
    PSI: np.array
        converted 2D array
    """
    vindex = 0
    PSI = np.zeros([Ny, Nx], dtype='complex')
    for i in range(Ny):
        for j in range(Nx):
            PSI[i, j] = psi[vindex]
            vindex = vindex + 1 
    return PSI

def sho_wavefunctions_plot(xmin=-10, xmax=10, Nx=250,
                           ymin=-10, ymax=10, Ny=250,
                           params=[1,1], neigs=6, E0=10, findpsi=True):
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
                V[vindex] = params[0]*X[i] ** 2 + params[1]*Y[j] ** 2
                vindex = vindex + 1
        return V
    
    eval_wavefunctions(xmin,xmax,Nx,
                       ymin,ymax,Ny,
                       Vfun,params,neigs,E0,findpsi)

def stadium_wavefunctions_plot(R=1, L=2, V0=1e6, neigs=6, E0=500, Ny=250):
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
    print("Axis limits:",xmin, xmax, ymin, ymax)

    Nx = int(Ny * 2 * R / (2.0 * R + L))
    print("Nx, Ny:",Nx, Ny)
    
    def Vfun2D(X, Y, params):
        R = params[0] # stadium radius
        L = params[1] # stadium length
        V0 = params[2] # stadium wall potential
        # stadium potential function
        Nx = len(X)
        Ny = len(Y)
        [x, y] = np.meshgrid(X, Y)
        F = np.zeros([Ny, Nx])

        for i in range(Nx):
            for j in range(Ny):
                if abs(X[i]) == R or abs(Y[j]) == R + 0.5 * L:
                    F[j, i] = V0
                if (abs(Y[j]) - 0.5 * L) > 0 and np.sqrt((abs(Y[j]) - 0.5 * L) ** 2 + X[i] ** 2) >= R:
                    F[j, i] = V0
        # simplify the 2D matrix to a 1D array for faster processing:
        V = twoD_to_oneD(Nx, Ny, F)                
        return V
 
    eval_wavefunctions(xmin,xmax,Nx,
                       ymin,ymax,Ny,
                       Vfun2D,params,neigs,E0,findpsi=True)

def stadium_wavefunctions_3dplot(R=1, L=0, V0=1e6, neigs=6, E0=70, Ny=250):
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
    print("Axis limits:",xmin, xmax, ymin, ymax)

    Nx = int(Ny * 2 * R / (2.0 * R + L))
    print("Nx, Ny:",Nx, Ny)
    
    def Vfun2D(X, Y, params):
        R = params[0] # stadium radius
        L = params[1] # stadium length
        V0 = params[2] # stadium wall potential
        # stadium potential function
        Nx = len(X)
        Ny = len(Y)
        [x, y] = np.meshgrid(X, Y)
        F = np.zeros([Ny, Nx])

        for i in range(Nx):
            for j in range(Ny):
                if abs(X[i]) == R or abs(Y[j]) == R + 0.5 * L:
                    F[j, i] = V0
                if (abs(Y[j]) - 0.5 * L) > 0 and np.sqrt((abs(Y[j]) - 0.5 * L) ** 2 + X[i] ** 2) >= R:
                    F[j, i] = V0
        # simplify the 2D matrix to a 1D array for faster processing:
        V = twoD_to_oneD(Nx, Ny, F)                
        return V
    
    H = schrodinger2D(xmin,xmax,Nx,ymin,ymax,Ny,Vfun2D,params,neigs,E0,True)
    evl = H[0] # eigenvalues
    indices = np.argsort(evl)
    print("Energy eigenvalues:")
    for i,j in enumerate(evl[indices]):
        print("{}: {:.2f}".format(i + 1, np.real(j)))
    evt = H[1] # eigenvectors

    # unpack the vector into 2 dimensions for plotting:
    for n in range(evt.shape[1]):
        psi = evt[:, n]  
        PSI = oneD_to_twoD(Nx, Ny, psi)
        PSI = np.abs(PSI)**2
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        X, Y = np.meshgrid(H[2], H[3])
        ax.plot_surface(X, Y , PSI, cmap='jet')
        ax.axis('off')
        plt.show()      
