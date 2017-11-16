from scipy import *
import matplotlib.pyplot as plt
from scipy import sparse  # this does not import the sparse linalg library, do it separately
from scipy.sparse import linalg as sla  # to prevent potential mix ups

#This script shows the Schrodinger Equation from Quantum Physics can be solved in 1 and 2 dimensional
#potential wells, and the results show the wavefunctions of a subatomic particle in the well.

# this function solves numerically the energy eigenvalues, eigen functions etc. for a potential Vfun along
# the x-axis from x-lower to x-upper. Be sure to set the limits of x correctly to encompass the potential.
def schrodinger1D(xmin, xmax, Nx, Vfun, neigs, E0=0.0, findpsi=False):
    # E0 is used for the sigma argument if one is using scipy version 0.10 and above. For scipy version
    # 0.90 and below, E0 is useless as the sparse eigenvalue calculator does not have a sigma argument
    # implemented yet.
    # for this code we are using Dirichlet Boundary Conditions
    x = linspace(xmin, xmax, Nx)  # x axis grid
    dx = x[1] - x[0]  # x axis step size
    # Obtain the potential function values:
    V = Vfun(x)
    # create the Hamiltonian Operator matrix:
    # H=sparse.lil_matrix(2*eye(Nx))
    H = sparse.eye(Nx, Nx, format='lil')  # is this better than the previous statement?
    # there are some bugs involved using sparse.eye; maybe using sparse.diags would be a better choice
    # for initialisation purposes, set lil format. Change it to csc later on
    for i in range(Nx - 1):
        H[i, i] = 2  # diagonal of the sparse matrix filled with 2
        H[i, i + 1] = -1  # 'lower and upper diagonals' filled with -1
        H[i + 1, i] = -1
    H[-1, -1] = 2  # force the last element of the matrix at the bottom right corner to be 2
    H = H / (dx ** 2)  # divide everything by the step size squared
    # Add the potential into the Hamiltonian Operator matrix:
    for i in range(Nx):
        H[i, i] = H[i, i] + V[i]  # note that the potentials are added only to the diagonal term
    # now the numerical Hamiltonian Operator is Complete after initialising the sparse matrix, 
    # convert it to csc format as the sparse linalg libraries are optimized for csc matrix format
    H = H.tocsc()
    # note that sigma arguments do not work for scipy versions of 0.9 or earlier. Need at least
    # scipy version 0.10 for sigma arguments to work. In this case use which='SM' instead which is
    # slower but still does the job for this scenario (well more or less)
    [evl, evt] = sla.eigs(H, k=neigs, which='SM')
    # for 0.10 users: [evl,evt]=sla.eigs(H,k=neigs,sigma=E0)
    # evl is an array containing the eigenvalues of the sparse matrix
    # evt is a matrix containing the corresponding eigen-vectors of the sparse matrix 
    # note that we need to normalize the wave functions! <x|x>=1 is compulsory!
    for i in range(neigs):
        evt[:, i] = evt[:, i] / sqrt(trapz(conj(evt[:, i]) * evt[:, i], x))
        # using trapz is more accurate than using the vector summation
        # use the real() function to remove the 0j element as the results are a + 0j, a is real
        evl = real(evl)  # again use real() to remove 0j from a +0j where a is real
    if findpsi == False:
        return evl
    else: 
        return evl, evt, x
         
# this function generates an array of potentials from an array of positions 
# I put a and b here so I can call schrodinger1D on its own. Presently I use A = 1 and B = 0
# to simulate the 1D SHO of which the eigen-energies are known: K=2*n+1 so I can see
# if the program is giving me the correct results
def Vfun(L):
    V = 1 * L ** 2
    return V

def double_well_energy_plot(A, Bvec, neighs=20, Nx=500):
    # If Vfun is defined within a function, the function calling Vfun first looks for the Vfun within itself
    # before looking for Vfun outside. If Vfun is found inside itself, it calls Vfun from inside.
    def Vfunc(L):
        V = A * L ** 2 + B * L ** 4
        return V
    En = array([0.0, 0.0, 0.0])  # for three values if En
    for i in range(len(Bvec)):  # run the function for all values inside Bvec
        B = Bvec[i]
        H = schrodinger1D(-10, 10, Nx, Vfunc, neighs, 0.0, False)  # obtain lowest eigenvalues for the potential   
        H = real(H)  # remove the 0j from the a + 0j result where a is real
        H = sort(H)  # sort from smallest to biggest
        print 'The Double Well Energies Are:'
        print H  # print H so I can see what is going on
        x = 0 * H + B  # creates a vector x of length H with values of B
        plt.plot(x, H, 'yo')  # plot E vs B
        # to obtain the energy eigenvalues of the shifted harmonic oscillators:
    
        # Potential for the numerical solution for the QSHO Energies:
        # Note: I was curious whether my analytical  solution was correct, so I decided to include
        # a small section of code to compute the numerical values of the oscillator eigen energies
        # for the sake of comparison with the analytical solutions.
        xo = sqrt(-A / (2.0 * B))
        def HER(L):  # harmonic oscillator on the right of the origin
            V = -A ** 2 / (4.0 * B) - 2 * A * (L - xo) ** 2
            return V
        HR = schrodinger1D(-xo - 10, xo + 10, 500, HER, 6, 0.0, False)
        HR = sort(HR)
        print 'The Numerical Harmonic Energies Are:'
        print HR
        xR = 0 * HR + B
        plt.plot(xR, HR, 'rx')

        # Obtain the analytical solution for the QSHO energies:
        V0 = -A ** 2 / (4.0 * B)
        for n in range(3):
            En[n] = (V0 + (n + 0.5) * sqrt(-8 * A))
        xn = 0 * En + B
        print 'The Analytic Harmonic Energies Are:'
        print En
        plt.plot(xn, En, 'g*')   
        # Comment: The analytic and numerical harmonic energies are the same discounting some numerical error
        # as they should be!   
    plt.xlabel('B')
    plt.ylabel('Energy')    
    plt.xlim(Bvec.min() / 2.0, Bvec.max() + Bvec.min() / 2.0)
    plt.show()

# this function generates 8 wave function plots in 4 sub plots for the double well. The eight wave functions
# plotted correspond to the lowest 8 energy eigenvalues.
def double_well_wavefunctions_plot(A, B, Nx=500):

    def Vfun(L):
        V = A * L ** 2 + B * L ** 4
        return V
    # obtain the 8 smallest values of eigen energies and eigen functions:
    H = schrodinger1D(-10, 10, Nx, Vfun, 8, 0.0, True)
    evt = H[1]  # extract the eigenvector matrix
    x = H[2]  # extract the x position array
    i = 0
    count = 1
    while i < 8:
        y1 = real(conj(evt[:, i]) * evt[:, i])  # obtain the probabilities from the wave function
        y2 = real(conj(evt[:, i + 1]) * evt[:, i + 1])
        # again use the real() function to remove the 0j from a+0j where a is real
        # check the wave functions are all normalized to 1:
        # print "The normalizations are: "
        # print trapz(y1,x), trapz(y2,x)
        plt.subplot(4, 1, count)  # 4 sub plots with 2 graphs per plot
        plt.plot(x, y1, 'r')
        plt.plot(x, y2, 'g')
        i = i + 2  # advance the wave function index two at a time since two graphs are plotted on one plot
        count = count + 1  # advance sub plot index to a maximum of 4 subplots
    plt.show()

# this function is displays the effects of the closure relation.
def delta_approx_plot(neigs):
    Nx = 500
    def Vfun(x):
        V = -0.5 * x ** 2 + 0.02 * x ** 4
        return V
    H = schrodinger1D(-10, 10, Nx, Vfun, neigs, 0.0, True)
    evt = H[1]  # extract the eigenvector matrix of dim: [nx,neigs]
    x = H[2]  # extract the x position vector
    f = 0 * x  # create an array of zeros with the length of x
    
    for i in range(neigs):
        psi0 = conj(evt[Nx / 2, i])  # choose the value closest to 0 i.e. at the middle of the array
        psix = psi0 * evt[:, i]
        f = f + psix  # sum over all values of N i.e. sum to neigs     

    plt.plot(x, real(f), 'b')
    plt.plot(x, imag(f), 'r')    
    plt.show()

# 2D potential function: this function creates a 1D array of a 2D potential dependent on x(i) and y(j). The function
# covers the entire grid of x(i) and y(j) into a single 1D array.
def Vfun2D(X, Y):
    Nx = len(X)
    Ny = len(Y)
    M = Nx * Ny
    V = zeros([M, 1])
    vindex = 0
    for i in range(Nx):
        for j in range(Ny):
            V[vindex] = X[i] ** 2 + Y[j] ** 2
            vindex = vindex + 1
    return V

# 2D schrodinger equation solver:    
def schrodinger2D(xmin, xmax, Nx, ymin, ymax, Ny, Vfun2D, neigs, E0=0.0, findpsi=False):
    x = linspace(xmin, xmax, Nx)  # x axis grid
    dx = x[1] - x[0]  # x axis step size
    y = linspace(ymin, ymax, Ny)
    dy = y[1] - y[0]
    M = Nx * Ny  # total matrix "dimension"
    # The Laplacian Matrix will be an M x M matrix
    # The Psi Column Vector will be a M x 1 array
    # The Potential Matrix is an M x M "identity" matrix i.e. only the diagonial is non-zerp. This matrix can be
    # compressed into a 1D array (pretty obvious)
    V = Vfun2D(x, y)

    # I will use the kronecker product method to create the final "2D" laplacian:
    # first create the x-laplacian:
    Hx = sparse.lil_matrix(2 * eye(Nx))
    for i in range(Nx - 1):
        Hx[i, i + 1] = -1
        Hx[i + 1, i] = -1
    Hx = Hx / (dx ** 2)
    # then create the y-laplacian:      
    Hy = sparse.lil_matrix(2 * eye(Ny))
    for i in range(Ny - 1):
        Hy[i, i + 1] = -1
        Hy[i + 1, i] = -1
    Hy = Hy / (dy ** 2)
    # use the kronecker product method to combine both x and y laplacians to form the 2D Laplacian:
    Ix = sparse.lil_matrix(eye(Nx))
    Iy = sparse.lil_matrix(eye(Ny))
    H = sparse.kron(Iy, Hx) + sparse.kron(Hy, Ix)  # this is the final "Laplacian" which is an M x M matrix; M=Nx x Ny
    # However note that this creates a bsr sparse matrix!!!!!!!!!! We need to add in the potential so no choice but to 
    # force a H.tolil() change.
    H = H.tolil()
    # The Potential is now added to the diagonal of the 2D Laplacian Matrix:
    for i in range(M):
        H[i, i] = H[i, i] + V[i]    

    H = H.tocsc()  
    # [evl,evt]=sla.eigs(H,k=neigs,which='SM')
    # for 0.10 users: 
    [evl, evt] = sla.eigs(H, k=neigs, sigma=E0)
    # Note: shape(H) = M x M, shape(evl) = neigs, shape(evt) = M x neigs
    # Very big matrices involved. Watch the computer run time!
            
    if findpsi == False:
        return evl
    else: 
        return evl, evt, x, y
    
def stadium_wavefunctions_plot(R, L, E0):
    
    # take the centre of stadium to be (0,0)
    xmin = -0.5 * L - R
    xmax = 0.5 * L + R
    ymin = -R
    ymax = R
    print xmin, xmax, ymin, ymax
    # Nx=500 #???
    Nx = 250
    # Ny=250 #???
    # currently I use the following to scale the number of elements in the y axis to that of the
    # x axis according to the respective lengths. Obviously I am assuming the element sizes are the
    # same. We can always modify this section to allow user input and what not.
    Ny = int32(Nx * 2 * R / (2.0 * R + L))
    print Nx, Ny
    neigs = 6
    M = Nx * Ny
    
    # define the potential function:
    # we have a rectangular grid encompassing the stadium. For values of x and y that lie outside the stadium we force
    # the value of 10000. Else the default value of 0 stays.
    def Vfun2D(X, Y):
        Nx = len(X)
        Ny = len(Y)
        [x, y] = meshgrid(X, Y)
        F = zeros([Ny, Nx])
        # F=ones([Ny,Nx])*-10000
        for i in range(Ny):
            for j in range(Nx):
                if abs(Y[i]) == R or abs(X[j]) == R + 0.5 * L:
                    F[i, j] = 10000
                if (abs(X[j]) - 0.5 * L) > 0 and sqrt((abs(X[j]) - 0.5 * L) ** 2 + Y[i] ** 2) >= R:
                    F[i, j] = 10000
        # plt.pcolormesh(F)
        # plt.colorbar()
        V = zeros([Nx * Ny, 1])
        vindex = 0
        for i in range(Ny):
            for j in range(Nx):
                V[vindex] = F[i, j]
                vindex = vindex + 1                      
        """
        M=Nx*Ny
        V=zeros([M,1])
        vindex=0
        for i in range(Nx):
            for j in range(Ny):
                if abs(Y[j]) == R or abs(X[i]) == R+0.5*L:
                    V[vindex]=10000
                if (abs(X[i])-0.5*L) > 0 and sqrt((abs(X[i])-0.5*L)**2+Y[j]**2) >= R:
                    V[vindex]=10000          
                vindex=vindex+1
        """        
        return V
    
    # Plot the potential for easier checking purposes:
    V = Vfun2D(linspace(xmin, xmax, Nx), linspace(ymin, ymax, Ny))
    F = zeros([Ny, Nx])  #The entire 2D mesh has Ny rows and Nx columns!!!!
    vindex = 0
    for i in range(Ny):
        for j in range(Nx):
            F[i, j] = V[vindex]  # note that this matrix is "upside down" and we need to flip it up-down to get the
            # y-dimensions correct. x dimensions are originally correct and need not be corrected.
            vindex = vindex + 1
    F = flipud(F)
    plt.figure()
    plt.pcolormesh(F)
    plt.axis('equal')
    plt.title('Graphical Image of the 2D Potential Well')
    plt.colorbar()
    
    H = schrodinger2D(xmin, xmax, Nx, ymin, ymax, Ny, Vfun2D, neigs, E0, findpsi=True)
    evt = H[1]  # extract the eigen vector matrix
    # xpos=H[2] #extract the x position matrix
    # ypos=H[3] #extract the y position matrix
    plt.figure()
    G = zeros([Ny, Nx])  #The entire 2D mesh has Ny rows and Nx columns!!!!
    for n in range(6):
        psi = evt[:, n]  # extract all the 6 eigen functions to plot one by one. Each eigen vector is a column array
        vindex = 0
        # size of psi = 10000 x 1
        for i in range(Ny):
            for j in range(Nx):
                G[i, j] = psi[vindex]
                vindex = vindex + 1
        G = flipud(G)        
        plt.subplot(2, 3, n + 1)    
        plt.pcolormesh(G)
        plt.axis('equal')
        plt.suptitle('Various Possible Quantum States in a 2D Potential Well')
    
    plt.show()
    
##########################################################################################################

# Additional code for Eclipse IDE users to run the functions:

#H=schrodinger1D(-10,10,500,Vfun,10,0.0,False)
#print H
"""
print H[0]
E=H[1]
x=H[2]
plt.plot(x,abs(E[:,3])**2,'r')
plt.show()
"""
"""
Comments:
If findpsi is false:
H=eigenvalues
If findpsi is true:
H[0] = eigenvalues
H[1] = eigenvectors matrix, dim: [nx,neigs]
H[2] = array of x-axis positions
"""

#double_well_energy_plot(-0.5, array([0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]))
"""
comments: the correspondence between the SHO energies and the double well energies are very close
for very low energies i.e. the first few energies. Also the result of the energies of the SHO
for both the numerical solution and the analytical solution are essentially the same within 
numerical accuracy so we can conclude that the results are OK.
"""

#double_well_wavefunctions_plot(-0.5,0.01)
"""
comments: to return the wave functions for the harmonic oscillator simply set B = 0 and set A positive. This
can be used to check if the results are correct.
"""

#delta_approx_plot(200)
"""
comments: as N increases, the peak formed becomes narrower and sharper and taller, so when N tends to
infinity, a Dirac Delta Function will be the result. This is an example of the closure relation.
"""

#schrodinger2D(-1,1,100,-1,1,100,Vfun2D,6,E0=0.0,findpsi=False)
"""
comments:this code takes a very long time to run!
"""

stadium_wavefunctions_plot(1.0,2.0,1000.0)
"""
comments: note this code takes forever to run if using scipy 0.9. Tested this code with a computer using scipy 0.10 and
it works. Using the sigma=E0 argument makes the code run faster. Tried it with various values of L
as well as E0. Very interesting graphical results. 
"""
