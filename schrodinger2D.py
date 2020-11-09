import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse import linalg as sla
 
def schrodinger2D(xmin, xmax, Nx, ymin, ymax, Ny,
                  Vfun2D, params, neigs, E0=0.0, findpsi=False):
    x = np.linspace(xmin, xmax, Nx)  
    dx = x[1] - x[0]  
    y = np.linspace(ymin, ymax, Ny)
    dy = y[1] - y[0]

    V = Vfun2D(x, y, params)

    # create the 2D Hamiltonian matrix
    Hx = sparse.lil_matrix(2 * np.eye(Nx))
    for i in range(Nx - 1):
        Hx[i, i + 1] = -1
        Hx[i + 1, i] = -1
    Hx = Hx / (dx ** 2)
    
    Hy = sparse.lil_matrix(2 * np.eye(Ny))
    for i in range(Ny - 1):
        Hy[i, i + 1] = -1
        Hy[i + 1, i] = -1
    Hy = Hy / (dy ** 2)

    Ix = sparse.lil_matrix(np.eye(Nx))
    Iy = sparse.lil_matrix(np.eye(Ny))
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
    
def eval_wavefunctions(xmin, xmax, Nx,
                       ymin, ymax, Ny,
                       Vfun, params, neigs, E0, findpsi):
        
    H = schrodinger2D(xmin,xmax,Nx,ymin,ymax,Ny,Vfun,params,neigs,E0,findpsi)
    evl = H[0] # eigenvalues
    indices = np.argsort(evl)
    print("Energy eigenvalues:")
    for i,j in enumerate(evl[indices]):
        print("{}: {:.2f}".format(i + 1, np.real(j)))
    evt = H[1] # eigenvectors
    plt.figure(figsize=(8, 8))
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
    # from a 2D matrix F return a 1D vector V
    V = np.zeros(Nx * Ny)
    vindex = 0
    for i in range(Ny):
        for j in range(Nx):
            V[vindex] = F[i, j]
            vindex = vindex + 1                      
    return V

def oneD_to_twoD(Nx, Ny, psi):
    # from a 1D vector psi return a 2D matrix PSI
    vindex = 0
    PSI = np.zeros([Ny, Nx], dtype='complex')
    for i in range(Ny):
        for j in range(Nx):
            PSI[i, j] = psi[vindex]
            vindex = vindex + 1 
    return PSI

def sho_wavefunctions_plot(xmin=-10, xmax=10, Nx=250,
                           ymin=-10, ymax=10, Ny=250,
                           params=[1,1], neigs=8, E0=10, findpsi=True):
    
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

def stadium_wavefunctions_plot(R=1, L=2, V0=1e6, neigs=6, E0=500):
    # R = stadium radius
    # L = stadium length
    # V0 = stadium wall potential
    ymin = -0.5 * L - R
    ymax = 0.5 * L + R
    xmin = -R
    xmax = R
    params = [R, L, V0]
    print("Axis limits:",xmin, xmax, ymin, ymax)

    Ny = 250
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