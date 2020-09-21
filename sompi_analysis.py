#for reference this line is exactly 80 characters long#########################
from scipy import randn
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from scipy import signal

#This code demonstrates the Sompi algorithm used to decompose a signal into
#both real and imaginary frequency parts. In contrast to traditional Fourier
#analysis techniques the Sompi method is able to give a direct interpretation
#of decaying/amplifying waveforms.

#BASED ON KUMAZAWA et al. 1990, A theory of spectral analysis based on the characteristic property of a linear dynamic system

#TEST WAVEFORM:
dt=1
T=np.arange(0,200,dt)
EFF=0.1
GEE=-0.003
X=np.sin(EFF*2*np.pi*T)*np.exp(GEE*2*np.pi*T)+randn(len(T))*0.03

def sompi(MORDERMIN=4,MORDERMAX=40,DMORDER=1):
    
    N_SOLUTIONS=int((MORDERMAX-MORDERMIN)/DMORDER+1)
    
    F=[[]]*N_SOLUTIONS #list to hold the solutions
    G=[[]]*N_SOLUTIONS #list to hold the solutions
    
    NORDER=len(X) #total number of data points in the time series
    count=0 #index holder for F and G
    for MORDER in range(MORDERMIN,MORDERMAX+DMORDER,DMORDER):
        P=np.zeros([MORDER+1,MORDER+1]) #P(k,l) matrix
        for k in range(0,MORDER+1,1):
            for l in range(0,MORDER+1,1):
                for t in range(MORDER,NORDER,1):
                    P[k,l]=P[k,l]+X[t-k]*X[t-l]
        P=P/(NORDER-MORDER) #TAKE THE AVERAGE!
        [val,vct]=linalg.eig(P)
        val=np.real(val) #both val and vct should be real! Drop +0j parts
        #vct=real(vct) #vct should be real
        aj=vct[:,np.argmin(val)] #smallest eigen value is the noise power
        Z=np.roots(aj) #obtain the m independent roots
        giw=np.log(Z) #Z=exp(gamma+i*omega)
        g=np.real(giw)/(2*np.pi)/dt #compare with GEE for the test waveform
        f=np.imag(giw)/(2*np.pi)/dt #compare with EFF for the test waveform
        F[count]=f
        G[count]=g
        count=count+1
    return F, G

F,G = sompi()
#obtain the FFT as a double check
[f,p]=signal.periodogram(X,1,None,2**12)

plt.subplot(2,1,1)
for count in range(len(F)):
    plt.plot(F[count],G[count],'b+')

plt.plot([EFF],[GEE],'ro')
#plt.xlim([0,0.5])
#plt.ylim([-0.02,0])
plt.axis([0,0.5,GEE*2,0])
plt.grid('on')
#plt.xlabel('Frequency Hz')
#plt.ylabel('Growth rate 1/s')
plt.subplot(2,1,2)
plt.plot(f,abs(p))
plt.xlim([0,0.5])
plt.show()

