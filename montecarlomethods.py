from __future__ import division  # / float division, // integer division
from scipy import *
import matplotlib.pyplot as plt
from scipy import random
import time
random.seed() #seed the RNG.

#This code using the Monte-Carlo method to simulate the magnetic behavior of a 2 dimensional
#crystal. 2 methods are listed here.
#The first shows the Ising model solved using the Metropolis algorithm, and the second shows the
#Ising model solved using the Heatbath algorithm.

def ising_metropolis_2D(J, H, T, Nx, Ny, steps, warmup_steps):
    start = time.time()
    N = Nx * Ny  # number of spins
    k = 1  # boltzmann constant
    M = zeros(len(T))
 
    for t in range(len(T)):
        spin = ones(N)  # reset the spins for each temperature
        print 'Current temperature =', T[t]
        B = 1 / (k * T[t])
        pflip = zeros([2, 5])  # there are a total of 10 weight values, 2 for 1,-1 and 5 for -4,-2,0,2,4
        # for each value of T pre compute the weights so as to speed up the computing time:
        Si = 1
        Sj = -4
        for i in range(2):  # 2 rows
            for j in range(5):  # 5 columns
                pflip[i, j] = exp(2 * (H + J * Sj) * Si * -B)  # probability of flipping the spin
                Sj = Sj + 2  # for -4,-2,0,2,4
            Si = -1  # "reset" Si to -1 for the second row:
            Sj = -4  # reset Sj to -4 again
        # pflip should have the form:

        # now for the warm up steps:
        for n in range(warmup_steps):
            r = int32(random.random() * N)  # randomly choose a lattice site
            x = mod(r, Nx)  # x-coordinate of spin i.e. column index
            y = r // Nx  # use integer division since y is an integer i.e. row index
            # note that at the end of the day all the indices must be 1-d since the spins are in an array
            s0 = spin[r]
            s1 = spin[mod(x + 1, Nx) + y * Ny]  #           S4   
            s2 = spin[x + mod(y + 1, Ny) * Nx]  #        S3 S0 S1
            s3 = spin[mod(x - 1 + Nx, Nx) + y * Nx]  #      S2
            s4 = spin[x + mod(y - 1 + Ny, Ny) * Nx]
            neighbours = s1 + s2 + s3 + s4  # sum of all the neighbouring spins
            if s0 == 1:
                pfliprow = 0  # row 0 of pflip contains spins of 1
            elif s0 == -1:
                pfliprow = 1  # row 1 of pflip contains spins of -1
            if neighbours == -4:
                pflipcol = 0  # col 0 of pflip contains Sj = -4
            elif neighbours == -2:
                pflipcol = 1  # col 1 of pflip contains Sj = -2
            elif neighbours == 0:
                pflipcol = 2  # col 2 of pflip contains Sj = 0
            elif neighbours == 2:
                pflipcol = 3  # col 3 of pflip contains Sj = 2
            elif neighbours == 4:
                pflipcol = 4  # col 4 of pflip contains Sj = 4
            rand = random.random()  # Test against weightage
            if rand < pflip[pfliprow, pflipcol]:
                spin[r] = -spin[r]  # flip the spin
                
        # now for the actual MC:
        for n in range(steps):
            r = int32(random.random() * N)
            x = mod(r, Nx)
            y = r // Nx
            s0 = spin[r]
            s1 = spin[mod(x + 1, Nx) + y * Ny]
            s2 = spin[x + mod(y + 1, Ny) * Nx]
            s3 = spin[mod(x - 1 + Nx, Nx) + y * Nx]
            s4 = spin[x + mod(y - 1 + Ny, Ny) * Nx]
            neighbours = s1 + s2 + s3 + s4
            if s0 == 1:
                pfliprow = 0
            elif s0 == -1:
                pfliprow = 1
            if neighbours == -4:
                pflipcol = 0
            elif neighbours == -2:
                pflipcol = 1
            elif neighbours == 0:
                pflipcol = 2
            elif neighbours == 2:
                pflipcol = 3 
            elif neighbours == 4:
                pflipcol = 4
            rand = random.random()
            if rand < pflip[pfliprow, pflipcol]:
                spin[r] = -spin[r] 

            # together with the Monte Carlo steps, perform the "Measurements:"
            M[t] = M[t] + sum(spin) / N

        # take the average values of the measurements over all MC steps
        M[t] = abs(M[t] / steps)  # take only the absolute values of M
        end = time.time() - start
        print "Total time elapsed so far:", end
    
    plt.plot(T, M, '-o')
    plt.xlabel('T')
    plt.ylabel('M')
    plt.show()   
    return [T,M]

def ising_heatbath_2D(J, H, T, Nx, Ny, steps, warmup_steps):
    start = time.time()
    N = Nx * Ny  # number of spins
    k = 1  # boltzmann constant
    M = zeros(len(T))  # magnetization vector

    for t in range(len(T)):
        spin = ones(N)  # reset the spins to 1 for each temperature
        print 'Current temperature =', T[t]
        B = 1 / (k * T[t])
        pflip = zeros(5)  # there are a total of 5 weight values for -4,-2,0,2,4
        # precompute the weights so as to speed up the computing time:
        Sj = -4
        for j in range(5):
            Hprime = (H + J * Sj)
            pflip[j] = exp(Hprime * B) / (exp(Hprime * B) + exp(Hprime * -B))  # probability of flipping spin up
            Sj = Sj + 2
        # now for the warm up steps:
        for n in range(warmup_steps):
            r = int32(random.random() * N)  # random choose lattice site
            x = mod(r, Nx)  # x-coordinate of spin i.e. column index
            y = r // Nx  # use integer division since y is an integer i.e. row index
            # note that at the end of the day all the indices must be 1-d since the spins are in an array
            s1 = spin[mod(x + 1, Nx) + y * Ny]
            s2 = spin[x + mod(y + 1, Ny) * Nx]
            s3 = spin[mod(x - 1 + Nx, Nx) + y * Nx]
            s4 = spin[x + mod(y - 1 + Ny, Ny) * Nx]
            neighbours = s1 + s2 + s3 + s4  # sum of all the neighbouring spins
            if neighbours == -4:
                pflipcol = 0  # col 0 of pflip contains Sj = -4
            elif neighbours == -2:
                pflipcol = 1  # col 1 of pflip contains Sj = -2
            elif neighbours == 0:
                pflipcol = 2  # col 2 of pflip contains Sj = 0
            elif neighbours == 2:
                pflipcol = 3  # col 3 of pflip contains Sj = 2
            elif neighbours == 4:
                pflipcol = 4  # col 4 of pflip contains Sj = 4
            rand = random.random()  # Test against weightage
            if rand < pflip[pflipcol]:
                spin[r] = 1  # flip the spin down
            else:
                spin[r] = -1  # flip the spin up
                
        # now for the actual MC:
        for n in range(steps):
            r = int32(random.random() * N)  # random choose lattice site
            x = mod(r, Nx)  # x-coordinate of spin i.e. column index
            y = r // Nx  # use integer division since y is an integer i.e. row index
            # note that at the end of the day all the indices must be 1-d since the spins are in an array
            s1 = spin[mod(x + 1, Nx) + y * Ny]
            s2 = spin[x + mod(y + 1, Ny) * Nx]
            s3 = spin[mod(x - 1 + Nx, Nx) + y * Nx]
            s4 = spin[x + mod(y - 1 + Ny, Ny) * Nx]
            neighbours = s1 + s2 + s3 + s4 
            if neighbours == -4:
                pflipcol = 0 
            elif neighbours == -2:
                pflipcol = 1 
            elif neighbours == 0:
                pflipcol = 2 
            elif neighbours == 2:
                pflipcol = 3 
            elif neighbours == 4:
                pflipcol = 4 
            rand = random.random() 
            if rand < pflip[pflipcol]:
                spin[r] = 1
            else:
                spin[r] = -1

            # together with the Monte Carlo steps, perform the "Measurements:"
            M[t] = M[t] + sum(spin) / N

        # take the average values of the measurements over all MC steps
        M[t] = abs(M[t] / steps)
        end = time.time() - start
        print "Total time elapsed so far:", end
    
    plt.plot(T, M, '-o')
    plt.xlabel('T')
    plt.ylabel('M')
    plt.show()
    return [T,M]

#################################################################################################
#FUNCTION CALLS TO TEST THE FUNCTION:
[T,M]=ising_metropolis_2D(1.0,0.0,linspace(0.01,10,100),20,20,100000,100000)
#[T,M]=ising_heatbath_2D(1.0,0.0,linspace(0.01,10,100),20,20,100000,100000)
