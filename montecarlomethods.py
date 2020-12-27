import numpy as np
import matplotlib.pyplot as plt
import time

# FUNCTION CALLS TO TEST THE FUNCTION:

# 2D ising metropolis algorithm
# [T2,M2]=ising_metropolis_2D(1.0,0.0,np.linspace(0.01,5,50),20,20,100000,100000)

# 2D ising heatbath algorithm
# [TH2,MH2]=ising_heatbath_2D(1.0,0.0,np.linspace(0.01,5,50),20,20,100000,100000)

# ND ising metropolis algorithm
# [TX,MX]=ising_metropolis_ND(2,1,0,np.linspace(0.01,5,50),20,100000,100000)

# This script has been superseded with the parallel process version:
# https://github.com/natsunoyuki/Computational_Physics_in_Python/blob/master/montecarlo_parallel.py

def ising_metropolis_2D(J, H, T, Nx, Ny, steps, warmup_steps):
    """
    This code using the Monte-Carlo method to simulate the magnetic behavior of a 2 dimensional
    crystal. 2 methods are listed here.
    The first shows the Ising model solved using the Metropolis algorithm, and the second shows the
    Ising model solved using the Heatbath algorithm.
    
    Inputs
    ------
    J: int
        coupling strength
    H: float
        external magnetic field strength
    T: np.array
        np.array of temperatures to conduct the MC simulation over
    Nx: int
        number of spins along x
    Ny: int
        number of spins along y
    steps: int
        number of MC steps to take
    warmup_steps: int
        number of parallel processors to use
        
    Returns
    -------
    T: np.array
        temperatures
    M: np.array
        magnetization
    """
    start = time.time()
    N = Nx * Ny  # number of spins
    k = 1  # boltzmann constant
    M = np.zeros(len(T))
 
    for t in range(len(T)):
        spin = np.ones(N)  # reset the spins for each temperature
        print('Current temperature =', T[t])
        B = 1 / (k * T[t])
        pflip = np.zeros([2, 5])  # there are a total of 10 weight values, 2 for 1,-1 and 5 for -4,-2,0,2,4
        # for each value of T pre compute the weights so as to speed up the computing time:
        Si = 1
        Sj = -4
        for i in range(2):  # 2 rows
            for j in range(5):  # 5 columns
                pflip[i, j] = np.exp(2 * (H + J * Sj) * Si * -B)  # probability of flipping the spin
                Sj = Sj + 2  # for -4,-2,0,2,4
            Si = -1  # "reset" Si to -1 for the second row:
            Sj = -4  # reset Sj to -4 again
        # pflip should have the form:

        # now for the warm up steps:
        for n in range(warmup_steps):
            spin = ising2D(Nx, Ny, spin, pflip)
                
        # now for the actual MC:
        for n in range(steps):
            spin = ising2D(Nx, Ny, spin, pflip)

            # together with the Monte Carlo steps, perform the "Measurements:"
            M[t] = M[t] + sum(spin) / N

        # take the average values of the measurements over all MC steps
        M[t] = abs(M[t] / steps)  # take only the absolute values of M
    end = int(time.time() - start)
    print("Total time elapsed:", end)
    
    #plt.plot(T, M, '-o')
    #plt.xlabel('T')
    #plt.ylabel('M')
    #plt.show()   
    return [T,M]

def ising2D(Nx, Ny, spin, pflip):
    """
    2D ising model
    """
    N = Nx * Ny
    r = int(np.random.random() * N)
    x = np.mod(r, Nx)
    y = r // Nx
    s0 = spin[r]
    s1 = spin[np.mod(x + 1, Nx) + y * Ny]
    s2 = spin[x + np.mod(y + 1, Ny) * Nx]
    s3 = spin[np.mod(x - 1 + Nx, Nx) + y * Nx]
    s4 = spin[x + np.mod(y - 1 + Ny, Ny) * Nx]
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
    rand = np.random.random()
    if rand < pflip[pfliprow, pflipcol]:
        spin[r] = -spin[r] 
    return spin
    

def ising_metropolis_1D(J, H, T, N, steps, warmup_steps):
    """
    This code using the Monte-Carlo method to simulate the magnetic behavior of a 2 dimensional
    crystal. 2 methods are listed here.
    The first shows the Ising model solved using the Metropolis algorithm, and the second shows the
    Ising model solved using the Heatbath algorithm.
    
    Inputs
    ------
    J: int
        coupling strength
    H: float
        external magnetic field strength
    T: np.array
        np.array of temperatures to conduct the MC simulation over
    N: int
        number of spins on each side of the cubic crystal
    steps: int
        number of MC steps to take
    warmup_steps: int
        number of MC warmup steps to take
        
    Returns
    -------
    T: np.array
        temperatures
    M: np.array
        magnetization
    """
    start = time.time()
    k = 1  # boltzmann constant
    M = zeros(len(T))
 
    for t in range(len(T)):
        spin = np.ones(N)  # reset the spins for each temperature
        print('Current temperature =', T[t])
        B = 1 / (k * T[t])
        pflip = np.zeros([2, 3])  # there are a total of 10 weight values, 2 for 1,-1 and 5 for -2,0,2
        # for each value of T pre compute the weights so as to speed up the computing time:
        Si = 1
        Sj = -2
        for i in range(2):  
            for j in range(3):  
                pflip[i, j] = np.exp(2 * (H + J * Sj) * Si * -B)  # probability of flipping the spin
                Sj = Sj + 2  # for -2,0,2
            Si = -1  # "reset" Si to -1 for the second row:
            Sj = -2  # reset Sj to -4 again
        # pflip should have the form:

        # now for the warm up steps:
        for n in range(warmup_steps):
            spin = ising1D(N, spin, pflip)
                
        # now for the actual MC:
        for n in range(steps):
            spin = ising1D(N, spin, pflip)

            # together with the Monte Carlo steps, perform the "Measurements:"
            M[t] = M[t] + sum(spin) / N

        # take the average values of the measurements over all MC steps
        M[t] = abs(M[t] / steps)  # take only the absolute values of M
    end = int(time.time() - start)
    print("Total time elapsed:", end)
    
    #plt.plot(T, M, '-o')
    #plt.xlabel('T')
    #plt.ylabel('M')
    #plt.show()   
    return [T,M] 

def ising1D(N, spin, pflip):
    """
    1D ising model
    """
    r = int(np.random.random() * N)  # randomly choose a lattice site
    s0 = spin[r]
    s1 = spin[np.mod(r + 1, N)]  #     S2 S0 S1       
    s2 = spin[r - 1]
    neighbours = s1 + s2
    if s0 == 1:
        pfliprow = 0  # row 0 of pflip contains spins of 1
    elif s0 == -1:
        pfliprow = 1  # row 1 of pflip contains spins of -1
    if neighbours == -2:
        pflipcol = 0  # col 0 of pflip contains Sj = -4
    elif neighbours == 0:
        pflipcol = 1  # col 1 of pflip contains Sj = -2
    elif neighbours == 2:
        pflipcol = 2  # col 2 of pflip contains Sj = 0
    rand = np.random.random()  # Test against weightage
    if rand < pflip[pfliprow, pflipcol]:
        spin[r] = -spin[r]  # flip the spin
    return spin
                
def ising_heatbath_1D(J, H, T, N, steps, warmup_steps):
    start = time.time()
    k = 1  # boltzmann constant
    M = np.zeros(len(T))  # magnetization vector

    for t in range(len(T)):
        spin = np.ones(N)  # reset the spins to 1 for each temperature
        print('Current temperature =', T[t])
        B = 1 / (k * T[t])
        pflip = np.zeros(3)  # there are a total of 3 weight values for -2,0,2
        # precompute the weights so as to speed up the computing time:
        Sj = -2
        for j in range(3):
            Hprime = (H + J * Sj)
            pflip[j] = np.exp(Hprime * B) / (np.exp(Hprime * B) + np.exp(Hprime * -B))  # probability of flipping spin up
            Sj = Sj + 2
        # now for the warm up steps:
        for n in range(warmup_steps):
            r = int(np.random.random() * N)  # randomly choose a lattice site

            s0 = spin[r]
            s1 = spin[np.mod(r + 1, N)]  #     S2 S0 S1       
            s2 = spin[r - 1]
         
            neighbours = s1 + s2
            if neighbours == -2:
                pflipcol = 0  # col 0 of pflip contains Sj = -4
            elif neighbours == 0:
                pflipcol = 1  # col 1 of pflip contains Sj = -2
            elif neighbours == 2:
                pflipcol = 2  # col 2 of pflip contains Sj = 0
            rand = np.random.random()  # Test against weightage
            if rand < pflip[pflipcol]:
                spin[r] = 1  # flip the spin down
            else:
                spin[r] = -1  # flip the spin up
                
        # now for the actual MC:
        for n in range(steps):
            r = int(np.random.random() * N)  # randomly choose a lattice site

            s0 = spin[r]
            s1 = spin[np.mod(r + 1, N)]  #     S2 S0 S1       
            s2 = spin[r - 1]
         
            neighbours = s1 + s2
            if neighbours == -2:
                pflipcol = 0  # col 0 of pflip contains Sj = -4
            elif neighbours == 0:
                pflipcol = 1  # col 1 of pflip contains Sj = -2
            elif neighbours == 2:
                pflipcol = 2  # col 2 of pflip contains Sj = 0
            rand = np.random.random()  # Test against weightage
            if rand < pflip[pflipcol]:
                spin[r] = 1  # flip the spin down
            else:
                spin[r] = -1  # flip the spin up

            # together with the Monte Carlo steps, perform the "Measurements:"
            M[t] = M[t] + sum(spin) / N

        # take the average values of the measurements over all MC steps
        M[t] = abs(M[t] / steps)
    end = int(time.time() - start)
    print("Total time elapsed:", end)
    
    #plt.plot(T, M, '-o')
    #plt.xlabel('T')
    #plt.ylabel('M')
    #plt.show()
    return [T, M]
    
def ising_heatbath_2D(J, H, T, Nx, Ny, steps, warmup_steps):
    start = time.time()
    N = Nx * Ny  # number of spins
    k = 1  # boltzmann constant
    M = np.zeros(len(T))  # magnetization vector

    for t in range(len(T)):
        spin = np.ones(N)  # reset the spins to 1 for each temperature
        print('Current temperature =', T[t])
        B = 1 / (k * T[t])
        pflip = np.zeros(5)  # there are a total of 5 weight values for -4,-2,0,2,4
        # precompute the weights so as to speed up the computing time:
        Sj = -4
        for j in range(5):
            Hprime = (H + J * Sj)
            pflip[j] = np.exp(Hprime * B) / (np.exp(Hprime * B) + np.exp(Hprime * -B))  # probability of flipping spin up
            Sj = Sj + 2
        # now for the warm up steps:
        for n in range(warmup_steps):
            r = int(np.random.random() * N)  # random choose lattice site
            x = np.mod(r, Nx)  # x-coordinate of spin i.e. column index
            y = r // Nx  # use integer division since y is an integer i.e. row index
            # note that at the end of the day all the indices must be 1-d since the spins are in an array
            s1 = spin[np.mod(x + 1, Nx) + y * Ny]
            s2 = spin[x + np.mod(y + 1, Ny) * Nx]
            s3 = spin[np.mod(x - 1 + Nx, Nx) + y * Nx]
            s4 = spin[x + np.mod(y - 1 + Ny, Ny) * Nx]
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
            rand = np.random.random()  # Test against weightage
            if rand < pflip[pflipcol]:
                spin[r] = 1  # flip the spin down
            else:
                spin[r] = -1  # flip the spin up
                
        # now for the actual MC:
        for n in range(steps):
            r = int(np.random.random() * N)  # random choose lattice site
            x = np.mod(r, Nx)  # x-coordinate of spin i.e. column index
            y = r // Nx  # use integer division since y is an integer i.e. row index
            # note that at the end of the day all the indices must be 1-d since the spins are in an array
            s1 = spin[np.mod(x + 1, Nx) + y * Ny]
            s2 = spin[x + np.mod(y + 1, Ny) * Nx]
            s3 = spin[np.mod(x - 1 + Nx, Nx) + y * Nx]
            s4 = spin[x + np.mod(y - 1 + Ny, Ny) * Nx]
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
            rand = np.random.random() 
            if rand < pflip[pflipcol]:
                spin[r] = 1
            else:
                spin[r] = -1

            # together with the Monte Carlo steps, perform the "Measurements:"
            M[t] = M[t] + sum(spin) / N

        # take the average values of the measurements over all MC steps
        M[t] = abs(M[t] / steps)
    end = int(time.time() - start)
    print("Total time elapsed:", end)
    
    #plt.plot(T, M, '-o')
    #plt.xlabel('T')
    #plt.ylabel('M')
    #plt.show()
    return [T, M]

def ising_metropolis_ND(D, J, H, T, Nx, steps, warmup_steps):
    #D = no. of dimensions
    #J = spin coupling parameter
    #H = external field
    #T = temperature
    #Nx = no. of spins along 1 axis (this code assumes cubic periodic crystal)
    #steps, warmp_steps = monte carlo steps
    start = time.time()
    N = Nx ** D
    k = 1
    M = np.zeros(len(T))
    if D == 1:
        print('One dimensional crystal')
        for t in range(len(T)):
            print('Current temperature:', t)
            spins = np.ones(Nx)
    elif D == 2:
        print('Two dimensional crystal')
        for t in range(len(T)):
            print('Current temperature:', T[t])
            spins = np.ones((Nx, Nx))
            B = 1 / (k * T[t])
            pflip = np.zeros([2, 5])  # there are a total of 10 weight values, 2 for 1,-1 and 5 for -4,-2,0,2,4
            # for each value of T pre compute the weights so as to speed up the computing time:
            Si = 1
            Sj = -4
            for i in range(2):  # 2 rows
                for j in range(5):  # 5 columns
                    pflip[i, j] = np.exp(2 * (H + J * Sj) * Si * -B)  # probability of flipping the spin
                    Sj = Sj + 2  # for -4,-2,0,2,4
                Si = -1  # "reset" Si to -1 for the second row:
                Sj = -4  # reset Sj to -4 again
            # pflip should have the form:
            for n in range(warmup_steps):
                x = int(np.random.random() * Nx)
                y = int(np.random.random() * Nx)
                s0 = spins[y, x]
                s1 = spins[y, np.mod(x+1, Nx)]
                s2 = spins[np.mod(y+1, Nx), x]
                s3 = spins[y, x - 1]
                s4 = spins[y - 1, x]            
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
                rand = np.random.random()
                if rand < pflip[pfliprow, pflipcol]:
                    spins[y, x] = -spins[y, x]  
            for n in range(steps):
                x = int(np.random.random() * Nx)
                y = int(np.random.random() * Nx)
                s0 = spins[y, x]
                s1 = spins[y, np.mod(x+1, Nx)]
                s2 = spins[np.mod(y+1, Nx), x]
                s3 = spins[y, x - 1]
                s4 = spins[y - 1, x]            
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
                rand = np.random.random()
                if rand < pflip[pfliprow, pflipcol]:
                    spins[y, x] = -spins[y, x]                  
                # together with the Monte Carlo steps, perform the "Measurements:"
                M[t] = M[t] + sum(spins) / N
            # take the average values of the measurements over all MC steps
            M[t] = abs(M[t] / steps)
        end = int(time.time() - start)
        print("Total time elapsed:", end)                
        return T, M
    elif D == 3:
        print('Three dimensional crystal')
        for t in range(len(T)):
            print('Current temperature:', T[t])
            spins = np.ones((Nx,Nx,Nx))
            B = 1 / (k * T[t])
            pflip = np.zeros([2, 7])  # there are a total of 10 weight values, 2 for 1,-1 and 7 for -6,-4,-2,0,2,4,6
            # for each value of T pre compute the weights so as to speed up the computing time:
            Si = 1
            Sj = -6
            for i in range(2):
                for j in range(7):
                    pflip[i, j] = np.exp(2 * (H + J * Sj) * Si * -B)  # probability of flipping the spin
                    Sj = Sj + 2  
                Si = -1  # "reset" Si
                Sj = -6  # reset Sj
            for n in range(warmup_steps):
                x = int(np.random.random() * Nx)
                y = int(np.random.random() * Nx)
                z = int(np.random.random() * Nx)
                s0 = spins[z, y, x]
                s1 = spins[z, y, np.mod(x + 1, Nx)]
                s2 = spins[z, np.mod(y + 1, Nx), x]
                s3 = spins[z, y, x - 1]
                s4 = spins[z, y - 1, x]
                s5 = spins[np.mod(z + 1, Nx), y, x]
                s6 = spins[z - 1, y, x]
                neighbours = s1 + s2 + s3 + s4 + s5 + s6
                if s0 == 1:
                    pfliprow = 0
                elif s0 == -1:
                    pfliprow = 1
                if neighbours == -6:
                    pflipcol = 0
                elif neighbours == -4:
                    pflipcol = 1
                elif neighbours == -2:
                    pflipcol = 2
                elif neighbours == 0:
                    pflipcol = 3
                elif neighbours == 2:
                    pflipcol = 4
                elif neighbours == 4:
                    pflipcol = 5
                elif neighbours == 6:
                    pflipcol = 6
                rand = np.random.random()
                if rand < pflip[pfliprow, pflipcol]:
                    spins[z, y, x] = -spins[z, y, x]
            for n in range(steps):
                x = int(np.random.random() * Nx)
                y = int(np.random.random() * Nx)
                z = int(np.random.random() * Nx)
                s0 = spins[z, y, x]
                s1 = spins[z, y, np.mod(x + 1, Nx)]
                s2 = spins[z, np.mod(y + 1, Nx), x]
                s3 = spins[z, y, x - 1]
                s4 = spins[z, y - 1, x]
                s5 = spins[np.mod(z + 1, Nx), y, x]
                s6 = spins[z - 1, y, x]
                neighbours = s1 + s2 + s3 + s4 + s5 + s6
                if s0 == 1:
                    pfliprow = 0
                elif s0 == -1:
                    pfliprow = 1
                if neighbours == -6:
                    pflipcol = 0
                elif neighbours == -4:
                    pflipcol = 1
                elif neighbours == -2:
                    pflipcol = 2
                elif neighbours == 0:
                    pflipcol = 3
                elif neighbours == 2:
                    pflipcol = 4
                elif neighbours == 4:
                    pflipcol = 5
                elif neighbours == 6:
                    pflipcol = 6
                rand = np.random.random()
                if rand < pflip[pfliprow, pflipcol]:
                    spins[z, y, x] = -spins[z, y, x]
                M[t] = M[t] + sum(spins) / N
            M[t] = abs(M[t] / steps)  # take only the absolute values of M
        print("Total time elapsed:", time.time()-start)
        return T, M
    else:
        print('Please choose a value of D = 1, 2 3...')
        return

    
def ising_metropolis_3D(J, H, T, Nx, Ny, Nz, steps, warmup_steps):
    start = time.time()
    N = Nx * Ny * Nz # number of spins
    k = 1  # boltzmann constant
    M = np.zeros(len(T))
 
    for t in range(len(T)):
        spin = np.ones(N)  # reset the spins for each temperature
        print('Current temperature =', T[t])
        B = 1 / (k * T[t])
        pflip = np.zeros([2, 7])  # there are a total of 10 weight values, 2 for 1,-1 and 7 for -6,-4,-2,0,2,4,6
        # for each value of T pre compute the weights so as to speed up the computing time:
        Si = 1
        Sj = -6
        for i in range(2):
            for j in range(7):
                pflip[i, j] = np.exp(2 * (H + J * Sj) * Si * -B)  # probability of flipping the spin
                Sj = Sj + 2  
            Si = -1  # "reset" Si
            Sj = -6  # reset Sj
        # pflip should have the form:

        # now for the warm up steps:
        for n in range(warmup_steps):
            spin = ising3D(Nx,Ny,Nz,spin,pflip)
                
        # now for the actual MC:
        for n in range(steps):
            spin = ising3D(Nx,Ny,Nz,spin,pflip)

            # together with the Monte Carlo steps, perform the "Measurements:"
            M[t] = M[t] + sum(spin) / N

        # take the average values of the measurements over all MC steps
        M[t] = abs(M[t] / steps)  # take only the absolute values of M
    end = int(time.time() - start)
    print("Total time elapsed:", end)
    
    #plt.plot(T, M, '-o')
    #plt.xlabel('T')
    #plt.ylabel('M')
    #plt.show()   
    return [T,M]

def ising3D(Nx,Ny,Nz,spin,pflip):
    N = Nx * Ny * Nz
    r = int(np.random.random() * N)
    x = np.mod(r, Nx)
    y = np.mod(r // Nx, Ny)
    z = r // Nx // Ny
    s0 = spin[r]
    s1 = spin[np.mod(x + 1, Nx) + y * Ny + z * Ny * Nz]
    s2 = spin[x + np.mod(y + 1, Ny) * Nx + z * Ny * Nz]
    s3 = spin[np.mod(x - 1 + Nx, Nx) + y * Nx + z * Ny * Nz]
    s4 = spin[x + np.mod(y - 1 + Ny, Ny) * Nx + z * Ny * Nz]
    s5 = spin[x + y * Ny + np.mod(z - 1, Nz) * Ny * Nz]
    s6 = spin[x + y * Ny + np.mod(z + 1, Nz) * Ny * Nz]
    neighbours = s1 + s2 + s3 + s4 + s5 + s6
    if s0 == 1:
        pfliprow = 0
    elif s0 == -1:
        pfliprow = 1
    if neighbours == -6:
        pflipcol = 0
    elif neighbours == -4:
        pflipcol = 1
    elif neighbours == -2:
        pflipcol = 2
    elif neighbours == 0:
        pflipcol = 3 
    elif neighbours == 2:
        pflipcol = 4
    elif neighbours == 4:
        pflipcol = 5
    elif neighbours == 6:
        pflipcol = 6
    rand = np.random.random()
    if rand < pflip[pfliprow, pflipcol]:
        spin[r] = -spin[r] 
    return spin    
