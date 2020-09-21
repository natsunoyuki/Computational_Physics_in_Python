import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
np.random.seed()  # seed the random number generator

# This script explores several properties of random numbers and random matrices

# This function samples from the exponential distribution using only the
# random.random() random number generator
def exp_sampler(lambd):
    # note that random.random range: 0 <= r <= 1
    x = -1.0 / lambd * np.log(np.random.random() / lambd)
    # I use the inversion technique to obtain this equation which samples
    # correctly from the exponential distribution
    return x

# This function generates a histogram of multiple samples from the exponential
# distribution and plots the analytical solution on with the numeric solution
def exp_hist(lambd, nruns=10000, nbins=100):
    result = np.linspace(1, nruns, nruns)  # empty array of length nruns for results
    for i in range(nruns):
        result[i] = exp_sampler(lambd)
        # result[i]=random.exponential(1/float(lambd))
        # I used this as a self check tool to see if I am getting correct result
        
    # generate line plot of p(x)
    x = np.linspace(0, max(result), 1000)
    y = lambd * np.exp(-lambd * x)
    
    plt.figure(figsize=(10,5))
    plt.hist(result, nbins, facecolor='b', normed=1)
    plt.plot(x, y, 'r')
    plt.show()

# This function generates multple subplots of histograms of the mean of the
# independent random variables sampled from the exponenential distribution.
# The histograms are plotted together with the analytical solution
# to the central limit theorem
def expmu_hist(lambd, nruns=10000, nbins=100):
    # number of samples:
    N = np.array([5, 10, 20])
    # create a vector to hold the results
    result = np.zeros([len(N), nruns])
    for a in range(nruns):
        for b in range(len(N)):
            for c in range(N[b]):
                result[b][a] = result[b][a] + exp_sampler(lambd)
            result[b][a] = result[b][a] / float(N[b])  # prevent integer division
            
    # Plotting of the analytical result:
    x = np.linspace(-result.min(), result.max(), 1000)
    # this allows linspace to scale the values of x according to the lambd value
    mu = lambd ** -1  # analytical mean
    p = np.zeros([len(N), len(x)])
    
    plt.figure(figsize=(10,5))
    for i in range(len(N)):
        sig2 = 1.0 / (N[i] * lambd ** 2)  # analytical variance which depends on N
        p[i, :] = ((2 * np.pi * sig2) ** -0.5) * np.exp(-(x - mu) ** 2 / 2.0 * sig2 ** -1)
        plt.subplot(len(N), 1, i + 1)
        plt.hist(result[i, :], nbins, normed=1)
        plt.plot(x, p[i, :])
        
    plt.show()
    # comments: as N gets larger and larger, the distribution tends to a
    # Gaussian distribution with corresponding mean and variance. This is the
    # essence of the central limit theorem.

# This function generates the eigenvalues from the gaussian orthogonal ensemble
# random matrices:
def goe_eigs(N):
	A = np.random.randn(N, N)  # create a real N x N matrix whose elements
	# are filled with random numbers
	B = 2 ** -0.5 * (A + A.T)
	D = linalg.eigvals(B)  # D = eigenvalues vector
	D = np.real(D)
	# include real() to prevent certain warnings from popping up in python
	# because for some matrices very small complex eigvals are returned
	# e.g. ~ e-17j
	# I know that this matrix has to produce only real eigenvalues.
	return D

# This function generates the eigenvalues from the symmetric bernoulli ensemble
# random matrices:
def bernoulli_eigs(N):
	B = np.zeros([N, N])
	# The following fills B[i][j] and B[j][i] with either 1 or -1 with
	# equal probabilities. The resulting matrix is symmetric
	for i in range(N):
		for j in range(N):
			if i >= j & i != j:
				x = np.random.random()
				if x <= 0.5:
					B[i][j] = 1
					B[j][i] = 1
				else:
					B[i][j] = -1
					B[j][i] = -1
	D = linalg.eigvals(B)  # D = eigenvalues vector
	D = np.real(D)
	# include real() to prevent certain warnings from popping up in ipython
	# because for some matrices very small complex eigvals are returned
	# e.g. ~ e-17j	
	# This matrix provides only real eigenvalues. The real() is a safeguard
	# in the case of numerical errors resulting in stuff like ~ e-17j
	return D		

# This function generates the eigenvalues from the gaussian unitary ensemble
# random matrices:	
def gue_eigs(N):
	A = np.random.randn(N, N) + 1j * np.random.randn(N, N)  # A=U+iV
	B = (2 ** -1) * (A + np.conj(A.T)) 
	D = linalg.eigvals(B)  # D = eigenvalues vector
	D = np.real(D)
	# include real() to prevent certain warnings from popping up in ipython
	# because for some matrices very small complex eigvals are returned
	# e.g. ~ e-17j
	# I know that hermitian matrices have real eigenvalues
	return D
	
# This function generates the histogram for an arbitrary random matrix ensemble
def random_matrix_hist(N, fun, nruns=1000, nbins=50):
	b = np.zeros([nruns, N])  # create an array to hold all the results of 1000 runs
	for i in range(nruns):
		b[i, :] = fun(N)  # output vector is placed into one of the rows of the result matrix
	# I put the real() in the matrix eigval generators instead of here which may
	# appear as if the code is repeating because I prefer to have it more
	# stringent in the sense that all outputs of the eigval generators
	# should be purely real, with no stuff like ~ e-17j
	bnorm = b * N ** -0.5  # obtain the normalised eigenvalues
	# Before passing this matrix of eigenvalues into hist(), we need to
	# reshape it into a vector:
	breshape = bnorm.reshape(np.size(bnorm), 1)  # reshape into column vector

	# To obtain eig spacing:
	DELTA = np.sort(bnorm)  # sort works row-wise i.e. sorts elements in each row only
	DELTA = np.diff(DELTA)  # diff works on the elements on each row only
	DELTA = DELTA.reshape(1, np.size(DELTA))  # reshape into a row vector
	mDELTA = np.mean(DELTA)  # obtain mean of DELTA i.e. eig spacing
	# I realise mean works faster on row rather than column vectors
	DELTA = DELTA * (mDELTA ** -1)  # divide by the mean value
	DELTA = DELTA.reshape(np.size(DELTA), 1)  # reshape back to column vector
	# for some reason hist does not accept row vectors as input, hence the need
	# for column vectors in the input of hist (at least on my computer)
	
	# PLOTTING:
	plt.subplot(2, 1, 1)  # two plots - Wigner's Semicircle and Surmise
	plt.hist(breshape, nbins, normed=1)
	x = np.linspace(-2, 2, 1000)
	pw = 1.0 * (2 * np.pi) ** -1 * (4 - x ** 2) ** 0.5  # Wigner Semi Circle Law
	plt.plot(x, pw, 'r')
	delta = np.linspace(0, 3, 1000)
	fw = np.pi * delta * 0.5 * np.exp((-np.pi * delta ** 2) * 0.25)  # Wigner Surmise
	# For the GUE ensemble, the following Surmise should be used instead:
	# c.f. Wigner surmise for mixed symmetry classes in random matrix theory
	# Sebastian Schierenberg, Falk Bruckmann, and Tilo Wettig
	# fw=32*pi**-2*delta**2*exp(-4*pi**-1*delta**2)
	plt.subplot(2, 1, 2)
	plt.hist(DELTA, nbins, normed=1)
	plt.plot(delta, fw, 'r')
	plt.ylim([0, 1])
	plt.xlim([0, 5])	
	plt.show()
