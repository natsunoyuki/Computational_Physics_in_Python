import numpy as np
from scipy import fft, ifft

#returns next power of 2 as fft algorithms work fastest when the input data
#length is a power of 2
def nextpow2(n):
    m_f = np.log2(n)
    m_i = np.ceil(m_f)
    return np.int(2 ** m_i)

#FFT function returns the frequency range up to Nyquist frequency
#and absolute spectral magnitude.
def FFT(y,t):
    dt=t[2]-t[1]
    Fs=1.0/dt
    L=len(y)
    Y=fft(y,L)*dt #dt should mathematically be included in the result!
    #amp=abs(Y)/(L/2) #FFT single sided spectrum
    amp=abs(Y) #or simply take the amplitude only?
    T=L*dt #1/T=Fs/L
    freq=np.arange(0,Fs/2,1/T) #list frequencies up to Nyquist frequency
    #resize result vectors to match their lengths
    if len(freq) < len(amp):
        amp = amp[0:len(freq)]  # make both vectors the same size
    elif len(amp) < len(freq):
        freq = freq[0:len(amp)]
    return freq,amp

#CEPSTRUM calculates the ceptram of a time series. The cepstrum is basically
#a fourier transform of a fourier transform and has units of time
def CEPSTRUM(y,t):
    dt=t[2]-t[1]
    #Fs=1.0/dt
    L=len(y)
    #Y=fft(y,L)
    #amp=np.abs(Y)/(L/2) #FFT single sided spectrum
    #T=L*dt #1/T=Fs/L
    #freq=np.arange(0,Fs/2,1/T) #list frequencies up to Nyquist frequency
    #C=real(ifft(log(abs(fft(y)))))
    C=np.abs(ifft(np.log(np.abs(fft(y))**2)))**2
    NumUniquePts=int(np.ceil((L+1)/2))
    C=C[0:NumUniquePts]
    q=np.arange(0,NumUniquePts,1)*dt
    return q,C

#these functions demonstrate explicitly the DFT algorithm, but should not 
#be used because of the extremely slow speed. Faster algorithms use the fact
#that FFT is symmetric.
def DFT(x):
    N=len(x)
    X=np.zeros(N,'complex')
    for k in range(0,N,1):
        for n in range(0,N,1):
            X[k]=X[k]+x[n]*np.exp(-1j*2*np.pi*k*n/N)
    return X

#inverse discrete transform
def IDFT(X):
    N=len(X)
    x=np.zeros(N,'complex')
    for n in range(0,N,1):
        for k in range(0,N,1):
            x[n]=x[n]+X[k]*np.exp(1j*2*np.pi*k*n/N)
    return x/N

#shifted inverse discrete transform
def SIDFT(X,D):
    N=len(X)
    x=np.zeros(N,'complex')
    for n in range(0,N,1):
        for k in range(0,N,1):
            x[n]=x[n]+np.exp(-1j*2*np.pi*k*D/N)*X[k]*np.exp(1j*2*np.pi*k*n/N)
    return x/N

#shifted fourier transform
def SHIFTFT(X,D):
    N=len(X)
    for k in range(N):
        X[k]=np.exp(-1j*2*np.pi*k*D/N)*X[k]
    return X
