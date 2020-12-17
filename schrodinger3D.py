import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse, linalg
from scipy.sparse import linalg as sla
 
def schrodinger3D(xmin, xmax, Nx, 
                  ymin, ymax, Ny, 
                  zmin, zmax, Nz, 
                  Vfun3D, params, neigs, E0=0.0, findpsi=False):
 
    x = np.linspace(xmin, xmax, Nx)  
    dx = x[1] - x[0]  
    y = np.linspace(ymin, ymax, Ny)
    dy = y[1] - y[0]
    z = np.linspace(zmin, zmax, Nz)
    dz = z[1] - z[0]

    V = Vfun3D(x, y, z, params)

    # create the 3D Hamiltonian matrix
    Hx = create_hamiltonian(Nx, dx)
    Hy = create_hamiltonian(Ny, dy)
    Hz = create_hamiltonian(Nz, dz)
    
    Ix = sparse.eye(Nx)
    Iy = sparse.eye(Ny)
    Iz = sparse.eye(Nz)
    
    # Combine the 3 individual 1 dimensional Hamiltonians
    # using Kronecker products
    Hxy = sparse.kron(Iy, Hx) + sparse.kron(Hy, Ix)
    Ixy = sparse.kron(Iy, Ix)
    H = sparse.kron(Iz, Hxy) + sparse.kron(Hz, Ixy)
    
    # Convert to lil form and add potential energy function
    H = H.tolil()
    for i in range(Nx * Ny * Nz):
        H[i, i] = H[i, i] + V[i]    

    # convert to csc form and solve the eigenvalue problem
    H = H.tocsc()  
    [evl, evt] = sla.eigs(H, k=neigs, sigma=E0)
            
    if findpsi == False:
        return evl
    else: 
        return evl, evt, x, y, z
    
def create_hamiltonian(Nx, dx):
    """
    This function creates a 1 dimensional Hamiltonian.
    """
    H = sparse.eye(Nx, Nx, format='lil') * 2
    for i in range(Nx - 1):
        H[i, i + 1] = -1
        H[i + 1, i] = -1
    H = H / (dx ** 2)  
    return H    

def sho_eigenenergies(xmin = -5, xmax = 5, Nx = 50, ymin = -5, ymax = 5, Ny = 50, 
                      zmin = -5, zmax = 5, Nz = 50, params = [1, 1, 1], neigs = 10, E0 = 0):
    """
    This function calculates the quantum simple harmonic oscillator eigenenergies.
    Theoretically, the eigenenergies are given by: E = hw(n + 3/2), n = nx + ny + nz.
    However, as we set h = w = 1, and we scale the energies during the Hamiltonian creation
    by 2, the theoretical eigenenergies are given by: E = 2n + 3.
    """
    def Vfun(X, Y, Z, params):
        """
        This function returns the potential energies for a 3D quantum harmonic oscillator.
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
                    V[vindex] = params[0]*X[i] ** 2 + params[1]*Y[j] ** 2 + params[2]*Z[k] ** 2
                    vindex = vindex + 1
        return V
    
    # Only eigenvalues will be returned!
    evl = schrodinger3D(xmin, xmax, Nx, ymin, ymax, Ny, zmin, zmax, Nz, Vfun, params, neigs, E0, False)
    
    indices = np.argsort(evl)
    print("Energy eigenvalues:")
    for i,j in enumerate(evl[indices]):
        print("{}: {:.2f}".format(i + 1, np.real(j)))
        
    return sorted(evl)
