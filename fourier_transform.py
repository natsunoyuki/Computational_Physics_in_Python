import numpy as np
from scipy import fft, ifft

def nextpow2(n):
    """
    returns next power of 2 as fft algorithms work fastest when the input data
    length is a power of 2
    
    Inputs
    ------
    n: int
        next power of 2 to calculate for
    
    Returns
    -------
    np.int(2 ** m_i): int
        next power of 2
    """
    m_f = np.log2(n)
    m_i = np.ceil(m_f)
    return np.int(2 ** m_i)

def FFT(y,t):
    """
    FFT function returns the frequency range up to Nyquist frequency
    and absolute spectral magnitude.
    
    Inputs
    ------
    y: np.array
        np.array of values to perform FFT
    t: np.array
        np.array of corresponding time values
        
    Returns
    -------
    freq: np.array
        np.array of frequency values
    amp: np.array
        np.array of Fourier amplitudes
    """
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

def CEPSTRUM(y,t):
    """
    CEPSTRUM calculates the ceptram of a time series. The cepstrum is basically
    a fourier transform of a fourier transform and has units of time
    
    Inputs
    ------
    y: np.array
        np.array of values to perform FFT
    t: np.array
        np.array of corresponding time values
        
    Returns
    -------
    q: np.array
        np.array of quefrency values
    C: np.array
        np.array of Cepstral amplitudes
    """
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

def DFT(x):
    """
    this function demonstrates explicitly the DFT algorithm, but should not 
    be used because of the extremely slow speed. Faster algorithms use the fact
    that FFT is symmetric.
    """
    N=len(x)
    X=np.zeros(N,'complex')
    for k in range(0,N,1):
        for n in range(0,N,1):
            X[k]=X[k]+x[n]*np.exp(-1j*2*np.pi*k*n/N)
    return X

def IDFT(X):
    """
    this function demonstrates explicitly the inverse DFT algorithm, but should not 
    be used because of the extremely slow speed. Faster algorithms use the fact
    that FFT is symmetric.
    """
    N=len(X)
    x=np.zeros(N,'complex')
    for n in range(0,N,1):
        for k in range(0,N,1):
            x[n]=x[n]+X[k]*np.exp(1j*2*np.pi*k*n/N)
    return x/N

def SIDFT(X,D):
    """
    this function demonstrates explicitly the shifted inverse DFT algorithm, but should not 
    be used because of the extremely slow speed. Faster algorithms use the fact
    that FFT is symmetric.
    """
    N=len(X)
    x=np.zeros(N,'complex')
    for n in range(0,N,1):
        for k in range(0,N,1):
            x[n]=x[n]+np.exp(-1j*2*np.pi*k*D/N)*X[k]*np.exp(1j*2*np.pi*k*n/N)
    return x/N

def SHIFTFT(X,D):
    """
    this function demonstrates explicitly the shifted DFT algorithm, but should not 
    be used because of the extremely slow speed. Faster algorithms use the fact
    that FFT is symmetric.
    """
    N=len(X)
    for k in range(N):
        X[k]=np.exp(-1j*2*np.pi*k*D/N)*X[k]
    return X
