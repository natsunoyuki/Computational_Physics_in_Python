import numpy as np
from numpy.lib.scimath import sqrt
import matplotlib.pyplot as plt
from scipy import linalg
from scipy import optimize

def transfer_matrix(E, V, L, V0):
    """
    This code demonstrates the concept of the transfer matrix method to calculate
    the transmission and reflection probabilities of a wave passing through
    different potentials. In this case we will utilize a qunatum particle.
    This code generates the transfer matrix for a series of quantum potentials
    
    WARNING! THIS CODE ASSUMES V0 IS THE SAME ON BOTH SIDES OF THE POTENTIAL
    WARNING! THIS CODE DOES NOT WORK IF E = V!!!
    The program will run only if V and L have the same length!
    
    Inputs
    ------
    E: float
        particle energy
    V: np.array
        np.array of potential energy steps
    L: np.array
        np.array of step lengths
    V0: float
        external potential
        
    Returns
    -------
    transfer: np.array
        transfer matrix
    """
    assert len(V) == len(L)
    N = len(L)
    V = np.append(V, V0)  # Append V0 to the end of V for usage in the for loop
    # calculate the array of k values:
    k = sqrt(E - V)  # note that k can be complex
              
    # calculate the first potential jump from V0 to V[0] i.e. from outside
    # to the first potential step using the combined matrix solved by hand:
    transfer = 0.5 * np.array([[1.0 + k[-1] / k[0], 1 - k[-1] / k[0]], [1 - k[-1] / k[0], 1 + k[-1] / k[0]]])
    # k[-1] is the last element in array k which is sqrt(E - V0) i.e. the value
    # of k in the outside region

    # Loop over the rest of the N steps starting with a translation 
    # across the space occupied by V1 i.e. along the length L1
    for n in range(N):
        # translate across V[n]:
        transl = np.array([[np.exp(1j * k[n] * L[n]), 0], [0, np.exp(-1j * k[n] * L[n])]])
        # jump from V[n] to V[n+1]:
        jump = 0.5 * np.array([[1.0 + k[n] / k[n + 1], 1 - k[n] / k[n + 1]], [1 - k[n] / k[n + 1], 1 + k[n] / k[n + 1]]])
        # now calculate the rest of the transfer matrix by "appending"
        # the other matrix multiplications infront of the current
        # transfer matrix:
        transfer = np.dot(transl, transfer)
        transfer = np.dot(jump, transfer)
        
    # return the completed transfer matrix:
    # check the determinant of the transfer matrix is 1
    # D=linalg.det(transfer)
    # print(abs(D)) #determinant of the transfer matrix should be 1
    return transfer

def barrier_plot():
    """
    This code generates the transmittance T and reflectance R for an incoming
    wave impinging onto a step potential:
    """
    E = np.linspace(1, 250, 700)  # range of E to be used
    V = np.array([50])  # various potential steps
    L = np.array([1])  # potential step lengths
    V0 = 0  # potential outside
    T = np.zeros(len(E))  # transmission results
    R = np.zeros(len(E))  # reflection results
    
    # sample over all values of E, except the case of E=V:
    for i in range(len(E)):
        M = transfer_matrix(E[i], V, L, V0)  # calculate transfer matrix for respective E
        R[i] = abs(M[1, 0] / M[1, 1]) ** 2  # calculate reflection
        T[i] = abs(linalg.det(M) / M[1, 1]) ** 2  # calculate transmission  
        
    # Eigen values of particle in a box are of the form:
    # H=n^2 hbar^2 pi^2 / 2 m L^2 + V
    # we have normalised hbar^2/2 m = 1:
    n = np.array([1, 2, 3, 4])
    H = (n ** 2) * (np.pi ** 2) + 50

    plt.figure(figsize=(15, 5))
    plt.semilogy(E, T, 'r')
    plt.semilogy(E, R, 'c')
    plt.vlines(H, 1e-7, 1)
    plt.legend(["Transmission", "Reflection", "Eigenvalues"])
    plt.xlabel('E')
    plt.ylabel('Transmission/Reflection Coefficients')
    plt.title('Potential Barrier Transmission and Reflections and Particle in a Box Eigenvalues')
    plt.show()

# Generally, the transmission curve increases from a miminum to a maximum
# transmission past the value of E=V=50, while the reflection remains at
# the maximum value and starts to decrease past the same value. For E<50
# this corresponds to the classical regime where the particle cannot
# overcome the potential barrier due to it having a lower energy than the
# potential barrier. Of course in QM the particle has a very small
# chance that it can tunnel through the barrier unlike in the classical
# regime where tunneling is impossible.

# The interesting feature is that past E=50 the reflection does not decay
# gradually i.e. the curve is not well behaved, there are some sharp dips
# in the curve at certain values of E, and these sharp dips occurs for the
# energies which are the same as those of energy eigenvalues of the particle
# in a box. Additionally, these sharp dips of the reflection curve coincide
# with a transmission value of 1 in the transmission curve; i.e. transmission
# is 1 only for certain values of E rather than for all values of E>V as in
# the classical case.

# T=1 only when the width of the barrier is half-integral or full-integral of
# the de Broglie wavelength of the particle within the Potential barrier:

#                     k*L = n*pi

# Working out the math leads to the analytical result that the energy of the
# particle follows the form of the energy eigenvalues of a particle in a box:
    
#                     E - V = n^2 * pi^2 / L^2

# This effect is a result of destructive interference between reflections of
# the waves at x=0 and x=L i.e. at the two edges of the potential barrier.

# See:
# Quantum Mechanics Vol 1, A. Messiah Pgs 88-98
# Quantum Mechanics, Bransden and Joachain Pgs 154-155
# Intro. to Quantum Mechanics, D. Griffiths Pg 82

def resonant_barrier_plot():
    """
    This code generates the transmission and reflection coefficients for a more
    complicated potential setup
    """
    V = np.array([250, 50, 250])
    L = np.array([0.2, 1, 0.2])
    E = np.linspace(1, 500, 700)
    V0 = 0
    T = np.zeros(len(E))
    R = np.zeros(len(E))

    for i in range(len(E)):
        M = transfer_matrix(E[i], V, L, V0)
        R[i] = abs(M[1, 0] / M[1, 1]) ** 2  # calculate Reflection
        T[i] = abs(linalg.det(M) / M[1, 1]) ** 2  # calculate Transmission

    plt.figure(figsize=(15, 5))
    plt.semilogy(E, T, 'r')
    plt.semilogy(E, R, 'c')
    plt.legend(["Transmission", "Reflection"])
    plt.xlabel('E')
    plt.ylabel('Transmission/Reflection Coefficients')
    plt.title('Potential Barrier Transmission and Reflections')
    plt.show()

# The transmission resonances i.e. transmission spikes in the transmission
# curve is similar in effect as observed for the barrier potential

# The potential pocket in the region -0.5 < x < 0.5 results in "resonance
# like tunnelling" (Resonance-like tunneling across a barrier with adjacent
# wells, S MAHADEVAN , P PREMA , S K AGARWALLA, B SAHU and C S SHASTRY) at
# specific particle energies which correspond to states of resonance within
# the potential pocket.

# i.e. what is happening is pseudo-standing waves are being formed within the
# two peaks of V=250 within the "valley" region where V=50 and this potential
# setup corresponds to a pseudo-bound state of a finite square well. These
# pseudo-standing waves occur only for certain pseudo-resonant states for
# certain energies E, at which transmission resonance occurs and the transmission
# spikes up to 1.

# As in the previous barrier_potential, during spikes in transmission, what is
# occuring is destructive interference occuring between reflections of the
# waves at the internal walls of the potential valley, and these resonances
# occur only when the width of the "well" is equal to an integral or half
# integral of the de Broglie wavelength of the particle inside the "well".

# See:
# Quantum Mechanics, Bransden and Joachain Pgs 169-170

# scipy.optimize.fsolve(fun,x0)
# calls fun(x), starts with x=x0 and tries to find a bunch of values near x0
# until it finds fun(x)=0 and return x

def find_bound_states(Emin, Emax, V, L, V0):
    """
    Find the bound states specified by Emin, Emax, V, L and V0
    
    Inputs
    ------
    Emin: float
        minimum energy
    Emax: float
        maximum energy
    V: np.array
        np.array of potential energies
    L: np.array
        np.array of the step lengths
    V0: float
        external potential energy value
    
    Returns
    -------
    R: np.array
        energies of the bound states
    """
    E = np.linspace(Emin, Emax, 700)
    H = np.array([])
    R = np.array([])

    def M22(E, V, L, V0):
        # this function returns the last element of the transfer matrix
        # i.e. M[1,1] in 0-index or M(2,2) in standard index
        Q = transfer_matrix(E, V, L, V0)[1, 1]
        return Q  # Q = M[1,1] i.e. Q is a float and not a matrix
    
    for i in range(len(E)):
        
        # obtain the optimized value of E which returns M22=0
        M = optimize.fsolve(M22, E[i], args=(V, L, V0))
        
        if Emin <= M  and M <= Emax and abs(M22(M, V, L, V0)) <= 1e-10 and M not in H:
            # For some stupid reason, optimize.fsolve was giving out nonsense
            # for certain values of E which were outside the range of Emin and
            # Emax, so restrict the solution of M only to within the range
            # allowed by Emin and Emax. Additionally some values of M output
            # by fsolve did not produce M22=0 so restrict only those values
            # which wil give M22=0. Also prevent duplicate entries of M being
            # input into H so as to make further filtering more efficient
            H = np.append(H, M)

    # Even with the first filter, still there remains elements which are
    # essentially duplicates of each other but with numeric differences of
    # order say 1e-14 etc. Hence we need to filter out these duplicates from
    # the final solution. I place this filter here because checking the value
    # of M and the elements in H within the for loop is not efficient, especially
    # when H is not yet sorted which makes filtering even harder. Better
    # to do the filtering outside the for loop:
    H = np.sort(H)
    R = np.append(R, H[0])
    for i in range(len(H) - 1):
        if abs(H[i + 1] - H[i]) >= 1e-10:
            R = np.append(R, H[i + 1])
    # return the final solution R which is an array containing the energies
    # of which boundstates occur for this particular potential
    return R

def resonant_barrier_plot2():
    V = np.array([250, 50, 250])
    L = np.array([0.2, 1, 0.2])
    E = np.linspace(1, 500, 700)
    V0 = 0
    T = np.zeros(len(E))
    R = np.zeros(len(E))
    # Obtain the transmission and reflection coefficients
    for i in range(len(E)):
        M = transfer_matrix(E[i], V, L, V0)
        R[i] = abs(M[1, 0] / M[1, 1]) ** 2  # calculate reflection
        T[i] = abs(linalg.det(M) / M[1, 1]) ** 2  # calculate transmission

    Q = find_bound_states(50, 250, np.array([50]), np.array([1]), 250)
    print('The bound states energies are: {}'.format(Q))
    
    plt.figure(figsize=(15, 5))
    plt.semilogy(E, T, 'r')
    plt.semilogy(E, R, 'c')
    plt.vlines(Q, 1e-13, 1)
    plt.legend(["Transmission", "Reflection", "Bound States"])
    plt.xlabel('E')
    plt.ylabel('Transmission/Reflection Coefficients')
    plt.title('Potential Barrier Transmission and Reflections')
    plt.show()

# What is observed is that the transmission spikes correspond closely to the
# bound state energies of a finite square well which is normally solved
# numerically or graphically. This implies that the potential setup in problem
# 2 does indeed correspond to a pseudo-bound state related to the bound state
# of a finite square well.

#T=transfer_matrix(1,np.array([10]),np.array([1]),0)
#print(T)
#barrier_plot()
#resonant_barrier_plot()
#resonant_barrier_plot2()
