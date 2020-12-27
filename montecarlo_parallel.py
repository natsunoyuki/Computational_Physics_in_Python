import numpy as np
import matplotlib.pyplot as plt
from scipy import random
import time
random.seed() #seed the RNG.
import multiprocessing as mp

class Worker:
    def __init__(self,D,J,H,Nx,steps,warmup_steps,results):
        self.D = D # number of spatial dimensions. D = 2 or 3 only.
        self.J = J # coupling strength J
        self.H = H # external magnetic field H
        self.Nx = Nx # number of spins per dimension. We assume a cubic lattice
        self.steps = steps # number of MC steps
        self.warmup_steps = warmup_steps # number of MC warmup steps     
        self.N = Nx ** D # total number of spins in cubic lattice
        self.k = 1 # Boltzmann constant kB
        self.results = results # results queue
        
    def __call__(self, q):
        """
        Inputs
        ------
        q: mp.JoinableQueue
            JoinableQueue of temperatures
        """
        while True:
            t = q.get()
            if t is None:
                return
            
            print("Processing temperature: {}".format(t))
            m = 0
            spin = np.ones(self.N)
            B = 1.0 / (self.k * t)
            
            pflip = self.calc_pflip(B) 
                
            for n in range(self.warmup_steps):
                spin = self.isingmodel(spin, pflip)
            for n in range(self.steps):
                spin = self.isingmodel(spin, pflip)
                m = m + np.sum(spin) / self.N
            m = np.abs(m / self.steps)
            
            self.results.put([t, m])
            
            q.task_done()
            
    def calc_pflip(self, B):
        """
        Calculates the probability of spin flip

        Inputs
        ------
        B: float
            1/kT

        Returns
        -------
        pflip: float
            probability of spin flip
        """
        if self.D == 2:
            pflip = np.zeros([2, 5])
            Si = 1
            Sj = -4
            for i in range(2):  
                for j in range(5):  
                    pflip[i, j] = np.exp(2 * (self.H + self.J * Sj) * Si * -B)  
                    Sj = Sj + 2  
                Si = -1  
                Sj = -4  
        elif self.D == 3:
            pflip = np.zeros([2, 7])
            Si = 1
            Sj = -6
            for i in range(2):
                for j in range(7):
                    pflip[i, j] = np.exp(2 * (self.H + self.J * Sj) * Si * -B)
                    Sj = Sj + 2  
                Si = -1  # "reset" Si
                Sj = -6  # reset Sj
        return pflip
    
    def isingmodel(self, spin, pflip):
        """
        Wrapper function for the actual 2D or 3D Ising model.

        Inputs
        ------
        spin: np.array
            np.array of spins
        pflip: float
            Probability of spin flip
        """
        if self.D == 2:
            spin = self.ising2D(spin, pflip)
        elif self.D == 3:
            spin = self.ising3D(spin, pflip)
        return spin
    
    def ising2D(self, spin, pflip):
        """
        Inputs
        ------
        spin: np.array
            np.array of spins
        pflip: float
            Probability of spin flip
        """      
        N = self.Nx ** self.D
        Nx = self.Nx
        Ny = self.Nx
        r = int(random.random() * N)
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
        rand = random.random()
        if rand < pflip[pfliprow, pflipcol]:
            spin[r] = -spin[r] 
        return spin
    
    def ising3D(self, spin, pflip):
        """
        Inputs
        ------
        spin: np.array
            np.array of spins
        pflip: float
            Probability of spin flip
        """
        N = self.N
        Nx = self.Nx
        Ny = self.Nx
        Nz = self.Nx
        r = int(random.random() * N)
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
        rand = random.random()
        if rand < pflip[pfliprow, pflipcol]:
            spin[r] = -spin[r] 
        return spin 
        
   
def main(D=2,J=1,H=0,T=np.linspace(0.01,5,50),Nx=20,steps=100000,nprocs=2):
    """
    Main driver function for the parallel process MC
    
    Inputs
    ------
    D: int
        number of spatial dimensions
    J: int
        coupling strength
    H: float
        external magnetic field strength
    T: np.array
        np.array of temperatures to conduct the MC simulation over
    Nx: int
        number of spins on each side of the cubic crystal
    steps: int
        number of MC steps to take
    nprocs: int
        number of parallel processors to use
        
    Returns
    -------
    T2: np.array
        temperatures
    M2: np.array
        magnetization
    """
    starttime = time.time()
    
    q = mp.JoinableQueue()
    for t in T:
        q.put(t)
        
    results = mp.Queue()
    processes = []
    for i in range(nprocs):
        worker=mp.Process(target=Worker(D,J,H,Nx,steps,steps,results),args=(q,),daemon=True)
        worker.start()
        processes.append(worker)
    q.join()
    
    T2 = []
    M2 = []
    for i in range(len(T)):
        t,m=results.get()
        T2.append(t)
        M2.append(m)
    T2 = np.array(T2)
    M2 = np.array(M2)    
    sorted_indices = np.argsort(T2)
    T2 = T2[sorted_indices]
    M2 = M2[sorted_indices]
    print("Total time elapsed: {:.3f}s".format(time.time()-starttime))
    
    return T2, M2
    
def demo():
    """
    This is a demo on how to perform MC simulations using the functions above.
    By default, the demo demonstrates the functions using a 2D lattice.
    """
    T, M = main(D=2, J=1, H=0, T=np.linspace(0.01,10,100), Nx=20, steps=100000, nprocs=2)
    plt.figure(figsize = (15, 5))
    plt.plot(T, M)
    plt.xlabel("Temperature")
    plt.ylabel("Magnetism")
    plt.show()
