import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sla
 

def schrodinger3D(xmin, xmax, Nx, 
                  ymin, ymax, Ny, 
                  zmin, zmax, Nz, 
                  Vfun3D, params, neigs, E0 = 0.0, findpsi = False):
    """
    Solves the 3 dimensional Schrodinger equation numerically
    
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
    zmin: float
        minimum value of the z axis
    zmax: float
        maximum value of the z axis
    Nz: int
        number of finite elements in the z axis           
    Vfun3D: function
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
    z: np.array
        z axis values
    """ 
    x = np.linspace(xmin, xmax, Nx)  
    dx = x[1] - x[0]  
    y = np.linspace(ymin, ymax, Ny)
    dy = y[1] - y[0]
    z = np.linspace(zmin, zmax, Nz)
    dz = z[1] - z[0]

    V = Vfun3D(x, y, z, params) # this is a 1D np.array

    # The following code might be problematic!
    """
    # Create the 3D Hamiltonian matrix.
    Hx = create_1d_hamiltonian(Nx, dx)
    Hy = create_1d_hamiltonian(Ny, dy)
    Hz = create_1d_hamiltonian(Nz, dz)
    
    Ix = sparse.eye(Nx)
    Iy = sparse.eye(Ny)
    Iz = sparse.eye(Nz)
    
    # Combine the 3 individual 1 dimensional Hamiltonians using Kronecker products
    Hxy = sparse.kron(Hx, Iy) + sparse.kron(Ix, Hy)
    Ixy = sparse.kron(Ix, Iy)
    H = sparse.kron(Hxy, Iz) + sparse.kron(Ixy, Hz)
    """
    # Use the following code instead!
    H = create_3d_hamiltonian(Nx, dx, Ny, dy, Nz, dz)
    
    # Convert to lil form and add potential energy function
    H = H.tolil()
    for i in range(Nx * Ny * Nz):
        H[i, i] = H[i, i] + V[i]    

    # convert to csc form and solve the eigenvalue problem
    H = H.tocsc()  
    [evl, evt] = sla.eigs(H, k = neigs, sigma = E0)
            
    if findpsi == False:
        return evl
    else: 
        return evl, evt, x, y, z
    

def create_1d_hamiltonian(Nx, dx):
    """
    This function creates a 1 dimensional Hamiltonian.
    
    Inputs
    ------
    Nx: int
        number of elements in the 1 spatial dimension
    dx: float
        step size of the 1 spatial dimension
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


def create_3d_hamiltonian(Nx, dx, Ny = None, dy = None, Nz = None, dz = None):
    """
    Creates a 3D Hamiltonian matrix.
    
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
    if Nz is None:
        Nz = Nx
    if dz is None:
        dz = dx
    
    Hxy = create_2d_hamiltonian(Nx, dx, Ny, dy)
    Iz = sparse.eye(Nz)
    Hz = create_1d_hamiltonian(Nx * Ny * Nz, 1)
    H = sparse.kron(Hxy, Iz) + Hz

    return H


def sho_eigenenergies(xmin = -5, xmax = 5, Nx = 50, 
                      ymin = -5, ymax = 5, Ny = 50, 
                      zmin = -5, zmax = 5, Nz = 50, 
                      params = [1, 1, 1], 
                      neigs = 10, E0 = 0):
    """
    This function calculates the quantum simple harmonic oscillator eigenenergies.
    Theoretically, the eigenenergies are given by: E = hw(n + 3/2), n = nx + ny + nz.
    However, as we set h = w = 1, and we scale the energies during the Hamiltonian creation
    by 2, the theoretical eigenenergies are given by: E = 2n + 3.

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
    zmin: float
        minimum value of the z axis
    zmax: float
        maximum value of the z axis
    Nz: int
        number of finite elements in the z axis           
    params: list
        list containing the parameters of Vfun
    neigs: int
        number of eigenvalues to find
    E0: float
        eigenenergy value to solve for

    Returns
    -------
    evl: list
        List of eigenenergies.
    """
    def Vfun(X, Y, Z, params):
        """
        This function returns the potential energies for a 3D quantum harmonic oscillator.
        
        Inputs
        ------
        X: np.array
            np.array of the x axis
        Y: np.array
            np.array of the y axis
        Z: np.array
            np.array of the z axis
        params: list
            list of parameters for the potential energy function
        
        Returns
        -------
        V: np.array
            np.array of the potential energy of the 3D QSHO
        """
        Nx = len(X)
        Ny = len(Y)
        Nz = len(Z)
        M = Nx * Ny * Nz
        V = np.zeros(M)
        vindex = 0
        for i in range(Nx):
            for j in range(Ny):
                for k in range(Nz):
                    V[vindex] = params[0]*X[i]**2 + params[1]*Y[j]**2 + params[2]*Z[k]**2
                    vindex = vindex + 1
        return V
    
    # Only eigenvalues will be returned!
    evl = schrodinger3D(xmin, xmax, Nx, ymin, ymax, Ny, zmin, zmax, Nz, Vfun, params, neigs, E0, False)
    
    indices = np.argsort(evl)
    print("Energy eigenvalues:")
    for i,j in enumerate(evl[indices]):
        print("{}: {:.2f}".format(i + 1, np.real(j)))
        
    return sorted(evl)


def hydrogen_eigenenergies(xmin = -10, xmax = 10, Nx = 101, 
                           ymin = -10, ymax = 10, Ny = 101, 
                           zmin = -10, zmax = -10, Nz = 101, 
                           params = [1 / np.pi / 4], 
                           neigs = 10, E0 = 0):
    def Vfun(X, Y, Z, params):
        Nx = len(X)
        Ny = len(Y)
        Nz = len(Z)
        M = Nx * Ny * Nz
        V = np.zeros(M)
        vindex = 0
        for i in range(Nx):
            for j in range(Ny):
                for k in range(Nz):
                    V[vindex] = params[0] / np.sqrt(X[i]**2 + Y[j]**2 + Z[k]**2)
                    vindex = vindex + 1
        return V

    evl = schrodinger3D(xmin, xmax, Nx, ymin, ymax, Ny, zmin, zmax, Nz, Vfun, params, neigs, E0, False)

    indices = np.argsort(evl)
    print("Energy eigenvalues:")
    for i,j in enumerate(evl[indices]):
        print("{}: {:.2f}".format(i + 1, np.real(j)))
        
    return sorted(evl)