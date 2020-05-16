import numpy as np
import matplotlib.pyplot as plt
from scipy import random
import time
random.seed() #seed the RNG.
import multiprocessing as mp

class Worker:
    def __init__(self,J,H,Nx,Ny,steps,warmup_steps,results):
        self.J = J
        self.H = H
        self.Nx = Nx
        self.Ny = Ny
        self.steps = steps
        self.warmup_steps = warmup_steps     
        self.N = Nx*Ny
        self.k = 1
        self.results=results
        
    def __call__(self,q):
        while True:
            t = q.get()
            if t is None:
                return
            
            print("Processing temperature: {}".format(t))
            m = 0
            spin = np.ones(self.N)
            B = 1/(self.k*t)
            pflip = np.zeros([2,5])
            Si = 1
            Sj = -4
            for i in range(2):  
                for j in range(5):  
                    pflip[i,j] = np.exp(2*(self.H+self.J*Sj)*Si*-B)  
                    Sj = Sj+2  
                Si = -1  
                Sj = -4              
            for n in range(self.warmup_steps):
                spin=self.ising2D(self.Nx,self.Ny,spin,pflip)
            for n in range(self.steps):
                spin=self.ising2D(self.Nx,self.Ny,spin,pflip)
                m = m + np.sum(spin)/self.N
            m = np.abs(m/self.steps)
            self.results.put([t,m])
            
            q.task_done()
            
    def ising2D(self,Nx,Ny,spin,pflip):
        N=Nx*Ny
        r = int(random.random()*N)
        x = np.mod(r,Nx)
        y = r//Nx
        s0 = spin[r]
        s1 = spin[np.mod(x+1,Nx)+y*Ny]
        s2 = spin[x+np.mod(y+1,Ny)*Nx]
        s3 = spin[np.mod(x-1+Nx,Nx)+y*Nx]
        s4 = spin[x+np.mod(y-1+Ny,Ny)*Nx]
        neighbours = s1+s2+s3+s4
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
        if rand < pflip[pfliprow,pflipcol]:
            spin[r] = -spin[r] 
        return spin
   
def main(J=1,H=0,T=np.linspace(0.01,5,50),N=20,steps=100000,nprocs=2):
    starttime = time.time()
    
    q = mp.JoinableQueue()
    for t in T:
        q.put(t)
        
    results = mp.Queue()
    processes = []
    for i in range(nprocs):
        worker=mp.Process(target=Worker(J,H,N,N,steps,steps,results),args=(q,),daemon=True)
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
    
    return T2,M2
    
T,M = main(J=1,H=0,T=np.linspace(0.01,5,50),N=20,steps=100000,nprocs=2)
plt.plot(T,M)
plt.xlabel("Temperature")
plt.ylabel("Magnetism")
plt.show()